"""Triage agent for diagnosis and treatment reasoning.

Combines symptom information and visual findings to generate diagnosis
and treatment recommendations via MedGemma. Uses Pydantic models and
JSON-based LLM output parsing.
"""

import logging
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from .config import Config

logger = logging.getLogger(__name__)


class Medication(BaseModel):
    """A single medication in a treatment plan."""

    name: str
    dosage: str
    duration: Optional[str] = None
    justification: Optional[str] = None


class Diagnosis(BaseModel):
    """Structured diagnosis result."""

    condition: str
    confidence: float
    supporting_evidence: List[str] = Field(default_factory=list)
    differential_diagnoses: List[str] = Field(default_factory=list)
    known_symptoms: List[str] = Field(default_factory=list)
    icd_code: Optional[str] = None


class TreatmentPlan(BaseModel):
    """Treatment plan for a diagnosis."""

    medications: List[Medication] = Field(default_factory=list)
    instructions: List[str] = Field(default_factory=list)
    follow_up_days: Optional[int] = None
    warning_signs: List[str] = Field(default_factory=list)
    requires_referral: bool = False
    referral_reason: Optional[str] = None


class ClinicalAssessment(BaseModel):
    """Schema the LLM must fill in. Included in the prompt via model_json_schema()."""

    condition: str = Field(description="Primary clinical condition or diagnosis")
    confidence: str = Field(description="Confidence level of the diagnosis (High, Medium, Low)")
    differential_diagnoses: List[str] = Field(default_factory=list, description="List of alternative diagnoses to consider")
    known_symptoms: List[str] = Field(default_factory=list, description="Symptoms observed in the patient")
    treatment: List[Medication] = Field(default_factory=list, description="Recommended medications and treatment plan")
    instructions: List[str] = Field(default_factory=list, description="Patient care instructions and management steps")
    warning_signs: List[str] = Field(default_factory=list, description="Critical warning signs that require immediate referral")
    follow_up_days: int = Field(default=3, description="Number of days until follow-up appointment")
    requires_referral: bool = Field(default=False, description="Whether patient needs referral to higher facility")
    referral_reason: Optional[str] = Field(default=None, description="Reason for referral if applicable")


CONFIDENCE_MAP = {"high": 0.85, "medium": 0.7, "low": 0.5}


class TriageAgent:
    """Medical reasoning agent for diagnosis and treatment generation.

    Delegates inference to an :class:`~healthpost.inference_backend.InferenceBackend`.
    """

    def __init__(self, config: Config, vision_analyzer=None, backend=None) -> None:
        self.config = config
        self.vision = vision_analyzer
        self._backend = backend

    @property
    def backend(self):
        """Lazily resolved inference backend."""
        if self._backend is None:
            from .inference_backend import create_backend
            self._backend = create_backend(self.config)
        return self._backend

    def _generate_response(self, prompt: str, max_tokens: int = 0) -> str:
        """Generate a text response via the inference backend."""
        tokens = max_tokens or self.config.max_new_tokens
        logger.debug(
            "Prompt length=%d, first 100 chars: %.100s", len(prompt), prompt,
        )
        return self.backend.generate_text(
            prompt,
            temperature=self.config.temperature,
            max_tokens=tokens,
        )

    def diagnose_and_treat(
        self,
        symptoms: str,
        visual_findings: List[str],
        patient_age: Optional[str] = None,
        current_medications: Optional[List[str]] = None,
    ) -> Tuple[Diagnosis, TreatmentPlan]:
        """Generate a diagnosis and treatment plan.

        Args:
            symptoms: Text description of symptoms.
            visual_findings: Findings from image analysis.
            patient_age: Age info (e.g. ``"adult"``, ``"child 5 years"``).
            current_medications: Patient's current medications.

        Returns:
            Tuple of ``(Diagnosis, TreatmentPlan)``.
        """
        logger.info(
            "diagnose_and_treat: symptoms=%.50s, visual_findings=%d, age=%s",
            symptoms, len(visual_findings), patient_age,
        )

        prompt = self._build_diagnosis_prompt(
            symptoms, visual_findings, patient_age, current_medications,
        )
        raw_json = self.backend.generate_structured(
            prompt,
            schema=ClinicalAssessment,
            temperature=self.config.temperature,
            max_tokens=1024,
        )
        try:
            assessment = ClinicalAssessment.model_validate_json(raw_json)
        except Exception as e:
            logger.warning("Structured output validation failed: %s", e)
            assessment = ClinicalAssessment(
                condition="Unknown condition", confidence="medium",
            )

        evidence = []
        if symptoms:
            evidence.append(f"Symptoms: {symptoms[:100]}")
        evidence.extend(visual_findings[:3])

        diagnosis = Diagnosis(
            condition=assessment.condition,
            confidence=CONFIDENCE_MAP.get(assessment.confidence.lower(), 0.7),
            supporting_evidence=evidence,
            differential_diagnoses=assessment.differential_diagnoses,
            known_symptoms=assessment.known_symptoms,
        )
        treatment = TreatmentPlan(
            medications=assessment.treatment,
            instructions=assessment.instructions,
            follow_up_days=assessment.follow_up_days,
            warning_signs=assessment.warning_signs,
            requires_referral=assessment.requires_referral,
            referral_reason=assessment.referral_reason,
        )

        logger.info(
            "diagnose_and_treat complete: condition=%s, confidence=%.2f, "
            "medications=%d, referral=%s",
            diagnosis.condition, diagnosis.confidence,
            len(treatment.medications), treatment.requires_referral,
        )
        return diagnosis, treatment

    def _build_diagnosis_prompt(
        self,
        symptoms: str,
        visual_findings: List[str],
        patient_age: Optional[str],
        current_medications: Optional[List[str]],
    ) -> str:
        """Build a diagnosis prompt. Schema enforcement is handled by Ollama."""
        prompt = (
            "You are a medical decision support system for a Community Health Worker "
            "in a rural health post. Analyze the patient case below and provide a "
            "complete clinical assessment.\n\n"
            "You MUST include:\n"
            "1. A primary diagnosis (condition) with confidence level\n"
            "2. Differential diagnoses to consider\n"
            "3. Known symptoms of the diagnosed condition\n"
            "4. A TREATMENT PLAN — if the condition requires medication, provide "
            "specific medications with drug name, dosage (e.g. '500mg twice daily'), "
            "duration (e.g. '5 days'), and a short justification for why this "
            "medication is needed. If no medications are needed, leave the medication "
            "list empty and focus on supportive care in the instructions.\n"
            "5. Patient care instructions\n"
            "6. Warning signs that require immediate referral\n"
            "7. Follow-up timeline\n"
            "8. Even when no medications are needed, the instructions MUST describe "
            "how to manage the condition (e.g. rest, warm compress, hydration) and "
            "the expected recovery timeline (e.g. 'symptoms should improve within 3-5 days').\n"
            "IMPORTANT: Do NOT leave instructions, warning_signs, known_symptoms, or "
            "differential_diagnoses empty. Always provide at least 2-3 items for each.\n\n"
            "PATIENT CASE:\n"
        )
        if patient_age:
            prompt += f"Age: {patient_age}\n"
        prompt += f"Symptoms: {symptoms}\n"
        if visual_findings:
            prompt += "Visual findings: " + "; ".join(visual_findings) + "\n"
        if current_medications:
            prompt += "Current medications: " + ", ".join(current_medications) + "\n"
        return prompt

