"""Triage agent for diagnosis and treatment reasoning.

Combines symptom information and visual findings to generate diagnosis
and treatment recommendations via MedGemma. Uses Pydantic models and
JSON-based LLM output parsing.
"""

import json
import logging
import re
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from .config import Config

logger = logging.getLogger(__name__)


class Medication(BaseModel):
    """A single medication in a treatment plan."""

    name: str
    dosage: str
    duration: Optional[str] = None


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

    condition: str
    confidence: str  # "high" / "medium" / "low"
    differential_diagnoses: List[str] = Field(default_factory=list)
    known_symptoms: List[str] = Field(default_factory=list)
    medications: List[Medication] = Field(default_factory=list)
    instructions: List[str] = Field(default_factory=list)
    warning_signs: List[str] = Field(default_factory=list)
    follow_up_days: int = 3
    requires_referral: bool = False
    referral_reason: Optional[str] = None


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
        response = self._generate_response(prompt, max_tokens=1024)
        assessment = self._parse_json_response(response)

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
            medications=assessment.medications,
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
        """Build a structured diagnosis prompt requesting JSON output."""
        schema = ClinicalAssessment.model_json_schema()

        prompt = (
            "You are a medical decision support system for a Community Health Worker.\n"
            "Analyze the patient case and respond with ONLY a valid JSON object.\n\n"
            "PATIENT CASE:\n"
        )
        if patient_age:
            prompt += f"Age: {patient_age}\n"
        prompt += f"Symptoms: {symptoms}\n"
        if visual_findings:
            prompt += "Visual findings: " + "; ".join(visual_findings) + "\n"
        if current_medications:
            prompt += "Current medications: " + ", ".join(current_medications) + "\n"
        prompt += (
            f"\nRespond with ONLY valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}\n\n"
            "JSON response:"
        )
        return prompt

    def _parse_json_response(self, response: str) -> ClinicalAssessment:
        """Extract JSON from LLM response, handle markdown fences and preamble."""
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", response)
        cleaned = cleaned.strip().rstrip("`")

        # Find the JSON object boundaries
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = cleaned[start:end]
            try:
                return ClinicalAssessment.model_validate_json(json_str)
            except Exception as e:
                logger.warning("JSON parse failed: %s", e)

        # Fallback: return defaults
        logger.warning("No valid JSON found in LLM response, using fallback")
        return ClinicalAssessment(condition="Unknown condition", confidence="medium")
