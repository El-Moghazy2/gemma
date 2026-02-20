"""Triage agent for diagnosis and treatment reasoning.

Combines symptom information and visual findings to generate diagnosis
and treatment recommendations via MedGemma 1.5.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class Medication:
    """A single medication in a treatment plan.

    Attributes:
        name: Drug name.
        dosage: Dosage description.
        duration: Treatment duration (e.g. ``"5 days"``).
        route: Administration route.
        frequency: Dosing frequency.
        notes: Additional prescribing notes.
    """

    name: str
    dosage: str
    duration: Optional[str] = None
    route: str = "oral"
    frequency: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class Diagnosis:
    """Structured diagnosis result.

    Attributes:
        condition: Primary diagnosis name.
        confidence: Confidence score between 0 and 1.
        supporting_evidence: Evidence supporting the diagnosis.
        differential_diagnoses: Alternative diagnoses considered.
        known_symptoms: Common symptoms of the diagnosed condition.
        icd_code: Optional ICD-10 code.
    """

    condition: str
    confidence: float
    supporting_evidence: List[str] = field(default_factory=list)
    differential_diagnoses: List[str] = field(default_factory=list)
    known_symptoms: List[str] = field(default_factory=list)
    icd_code: Optional[str] = None


@dataclass
class TreatmentPlan:
    """Treatment plan for a diagnosis.

    Attributes:
        medications: Prescribed medications.
        instructions: Non-medication instructions for the patient.
        follow_up_days: Recommended follow-up interval.
        warning_signs: Signs requiring immediate return.
        requires_referral: Whether hospital referral is needed.
        referral_reason: Explanation when referral is required.
    """

    medications: List[Medication] = field(default_factory=list)
    instructions: List[str] = field(default_factory=list)
    follow_up_days: Optional[int] = None
    warning_signs: List[str] = field(default_factory=list)
    requires_referral: bool = False
    referral_reason: Optional[str] = None


class TriageAgent:
    """Medical reasoning agent for diagnosis and treatment generation.

    Delegates inference to an :class:`~healthpost.inference_backend.InferenceBackend`.

    Attributes:
        config: Application configuration.
        vision: Optional vision analyzer for image-assisted triage.
        backend: Inference backend (Unsloth or HuggingFace).
    """

    def __init__(self, config: Config, vision_analyzer=None, backend=None) -> None:
        """Initialize the triage agent.

        Args:
            config: Application configuration.
            vision_analyzer: Optional ``MedicalVisionAnalyzer`` instance.
            backend: Inference backend. If ``None``, one will be created
                lazily via :func:`~healthpost.inference_backend.create_backend`.
        """
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
        """Generate a text response via the inference backend.

        Args:
            prompt: Full prompt text.
            max_tokens: Override for maximum tokens. Falls back to
                ``config.max_new_tokens`` when 0.

        Returns:
            Generated response string.
        """
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
        patient_conditions: Optional[List[str]] = None,
    ) -> Tuple[Diagnosis, TreatmentPlan]:
        """Generate a diagnosis and treatment plan.

        Args:
            symptoms: Text description of symptoms.
            visual_findings: Findings from image analysis.
            patient_age: Age info (e.g. ``"adult"``, ``"child 5 years"``).
            patient_conditions: Known pre-existing conditions.

        Returns:
            Tuple of ``(Diagnosis, TreatmentPlan)``.
        """
        logger.info(
            "diagnose_and_treat: symptoms=%.50s, visual_findings=%d, age=%s",
            symptoms, len(visual_findings), patient_age,
        )

        prompt = self._build_diagnosis_prompt(
            symptoms, visual_findings, patient_age, patient_conditions,
        )
        response = self._generate_response(prompt)

        diagnosis = self._parse_diagnosis(response, symptoms, visual_findings)
        treatment = self._parse_treatment(response, diagnosis)

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
        patient_conditions: Optional[List[str]],
    ) -> str:
        """Build a structured diagnosis prompt.

        Args:
            symptoms: Symptom description.
            visual_findings: Image-derived findings.
            patient_age: Patient age string.
            patient_conditions: Pre-existing conditions.

        Returns:
            Complete prompt requesting structured output.
        """
        prompt = (
            "You are a medical decision support system helping a "
            "Community Health Worker.\n"
            "Based on the patient information, provide diagnosis and "
            "treatment in the EXACT format shown below.\n\n"
            "PATIENT INFORMATION:\n"
        )
        if patient_age:
            prompt += f"Age: {patient_age}\n"
        if patient_conditions:
            prompt += f"Known conditions: {', '.join(patient_conditions)}\n"

        prompt += f"\nSYMPTOMS:\n{symptoms}\n"

        if visual_findings:
            prompt += "\nVISUAL FINDINGS:\n"
            for finding in visual_findings:
                prompt += f"- {finding}\n"

        prompt += (
            "\nRespond in EXACTLY this format:\n\n"
            "DIAGNOSIS: [condition name]\n"
            "CONFIDENCE: [high/medium/low]\n"
            "DIFFERENTIALS: [other possible conditions, comma separated]"
            "\n\n"
            "MEDICATIONS:\n"
            "- [Drug name]: [specific dosage and frequency]\n"
            "- [Drug name]: [specific dosage and frequency]\n\n"
            "INSTRUCTIONS:\n"
            "- [instruction 1]\n"
            "- [instruction 2]\n\n"
            "WARNING SIGNS:\n"
            "- [sign 1]\n"
            "- [sign 2]\n\n"
            "FOLLOW UP: [number] days\n"
            "REFERRAL: [yes/no] - [reason if yes]"
        )
        return prompt

    def _parse_diagnosis(
        self,
        response: str,
        symptoms: str,
        visual_findings: List[str],
    ) -> Diagnosis:
        """Parse structured model output into a ``Diagnosis``.

        Args:
            response: Raw model response.
            symptoms: Original symptom text (used as evidence).
            visual_findings: Image findings (used as evidence).

        Returns:
            Populated ``Diagnosis`` object.
        """
        condition = "Unknown condition"
        match = re.search(r"DIAGNOSIS:\s*(.+)", response, re.IGNORECASE)
        if match:
            condition = match.group(1).strip()
        else:
            logger.warning(
                "No DIAGNOSIS field found in response, falling back to "
                "'Unknown condition'",
            )

        confidence = 0.7
        match = re.search(
            r"CONFIDENCE:\s*(high|medium|low)", response, re.IGNORECASE,
        )
        if match:
            conf_map = {"high": 0.85, "medium": 0.7, "low": 0.5}
            confidence = conf_map.get(match.group(1).lower(), 0.7)

        differentials: List[str] = []
        match = re.search(
            r"DIFFERENTIALS:\s*(.+)", response, re.IGNORECASE,
        )
        if match:
            differentials = [
                d.strip() for d in match.group(1).split(",") if d.strip()
            ]

        evidence: List[str] = []
        if symptoms:
            evidence.append(f"Symptoms: {symptoms[:100]}")
        evidence.extend(visual_findings[:3])

        logger.debug(
            "Parsed diagnosis: condition=%s, confidence=%.2f, differentials=%d",
            condition, confidence, len(differentials),
        )
        return Diagnosis(
            condition=condition,
            confidence=confidence,
            supporting_evidence=evidence,
            differential_diagnoses=differentials,
        )

    def _parse_treatment(
        self, response: str, diagnosis: Diagnosis,
    ) -> TreatmentPlan:
        """Parse structured model output into a ``TreatmentPlan``.

        Args:
            response: Raw model response.
            diagnosis: Associated diagnosis (unused currently, reserved
                for future heuristics).

        Returns:
            Populated ``TreatmentPlan`` object.
        """
        medications = self._parse_medications_section(response)
        if not medications:
            logger.warning("No medications parsed from response")
        instructions = self._parse_list_section(response, "INSTRUCTIONS")
        warning_signs = self._parse_list_section(response, "WARNING SIGNS")

        follow_up = 3
        match = re.search(
            r"FOLLOW\s*UP:\s*(\d+)", response, re.IGNORECASE,
        )
        if match:
            follow_up = int(match.group(1))

        needs_referral = False
        referral_reason = None
        match = re.search(
            r"REFERRAL:\s*(yes|no)\s*[-:]?\s*(.*)",
            response, re.IGNORECASE,
        )
        if match:
            needs_referral = match.group(1).lower() == "yes"
            if needs_referral and match.group(2):
                referral_reason = match.group(2).strip()

        logger.debug(
            "Parsed treatment: medications=%d, follow_up_days=%d, referral=%s",
            len(medications), follow_up, needs_referral,
        )
        return TreatmentPlan(
            medications=medications,
            instructions=instructions,
            follow_up_days=follow_up,
            warning_signs=warning_signs,
            requires_referral=needs_referral,
            referral_reason=referral_reason,
        )

    def _parse_medications_section(self, response: str) -> List[Medication]:
        """Extract medications from the MEDICATIONS section.

        Args:
            response: Full model response.

        Returns:
            List of ``Medication`` objects.
        """
        medications: List[Medication] = []

        match = re.search(
            r"MEDICATIONS:\s*(.*?)"
            r"(?=\n(?:INSTRUCTIONS|WARNING|FOLLOW|REFERRAL|$))",
            response, re.IGNORECASE | re.DOTALL,
        )
        if not match:
            return medications

        for line in match.group(1).strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            line = re.sub(r"^[\-\*\u2022]\s*", "", line)

            if ":" in line or " - " in line:
                parts = re.split(r"[:\-]", line, 1)
                if len(parts) == 2:
                    name = parts[0].strip()
                    dosage = parts[1].strip()
                    if name and dosage:
                        medications.append(
                            Medication(name=name, dosage=dosage),
                        )
            elif line:
                medications.append(
                    Medication(name=line, dosage="as directed"),
                )

        return medications

    def _parse_list_section(
        self, response: str, section_name: str,
    ) -> List[str]:
        """Extract a bulleted list section from the model response.

        Args:
            response: Full model response.
            section_name: Header to search for (e.g. ``"INSTRUCTIONS"``).

        Returns:
            List of cleaned item strings.
        """
        items: List[str] = []
        pattern = (
            rf"{section_name}:\s*(.*?)"
            r"(?=\n(?:MEDICATIONS|INSTRUCTIONS|WARNING|FOLLOW|REFERRAL|$))"
        )
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)

        if match:
            for line in match.group(1).strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                line = re.sub(r"^[\-\*\u2022\d\.]+\s*", "", line).strip()
                if line:
                    items.append(line)
        return items

    def _extract_condition(self, response: str) -> str:
        """Extract primary diagnosis from an unstructured response.

        Args:
            response: Raw model response text.

        Returns:
            Extracted condition name.
        """
        response_lower = response.lower()
        for pattern in [
            r"diagnosis[:\s]+([^\n]+)",
            r"likely[:\s]+([^\n]+)",
        ]:
            match = re.search(pattern, response_lower)
            if match:
                condition = match.group(1).strip().split(".")[0]
                return condition.split(",")[0].title()

        common_conditions = [
            "malaria", "pneumonia", "diarrhea", "skin infection",
            "respiratory infection", "urinary tract infection",
            "measles", "ringworm", "wound infection", "dehydration",
            "fever", "gastroenteritis", "conjunctivitis",
            "otitis media",
        ]
        for condition in common_conditions:
            if condition in response_lower:
                return condition.title()

        return "Unspecified illness"

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence level from an unstructured response.

        Args:
            response: Raw model response text.

        Returns:
            Confidence score between 0 and 1.
        """
        response_lower = response.lower()

        if "high confidence" in response_lower or "confident" in response_lower:
            return 0.85
        if "medium confidence" in response_lower or "moderate" in response_lower:
            return 0.65
        if "low confidence" in response_lower or "uncertain" in response_lower:
            return 0.45

        match = re.search(r"(\d+)%", response)
        if match:
            return int(match.group(1)) / 100

        return 0.7

    def _extract_differentials(self, response: str) -> List[str]:
        """Extract differential diagnoses from an unstructured response.

        Args:
            response: Raw model response text.

        Returns:
            Up to 5 differential diagnosis names.
        """
        differentials: List[str] = []

        diff_section = re.search(
            r"differential[s]?[:\s]*(.*?)(?=\n\n|treatment|referral|$)",
            response.lower(), re.DOTALL,
        )
        if diff_section:
            items = re.split(r"[\n\-\*\d\.]+", diff_section.group(1))
            for item in items:
                item = item.strip()
                if item and len(item) > 3:
                    differentials.append(item.title())

        return differentials[:5]

    def _extract_medications(self, response: str) -> List[Medication]:
        """Extract medications from an unstructured response.

        Args:
            response: Raw model response text.

        Returns:
            List of ``Medication`` objects found in the response.
        """
        medications: List[Medication] = []
        found_meds: set = set()

        known_meds = {
            "paracetamol": "Paracetamol",
            "acetaminophen": "Paracetamol",
            "amoxicillin": "Amoxicillin",
            "metronidazole": "Metronidazole",
            "oral rehydration": "ORS",
            "ors": "ORS",
            "zinc": "Zinc",
            "vitamin a": "Vitamin A",
            "artemether": "Artemether-Lumefantrine",
            "lumefantrine": "Artemether-Lumefantrine",
            "coartem": "Artemether-Lumefantrine",
            "ibuprofen": "Ibuprofen",
            "clotrimazole": "Clotrimazole cream",
            "hydrocortisone": "Hydrocortisone cream",
            "cotrimoxazole": "Cotrimoxazole",
            "doxycycline": "Doxycycline",
            "azithromycin": "Azithromycin",
            "ciprofloxacin": "Ciprofloxacin",
            "prednisolone": "Prednisolone",
            "salbutamol": "Salbutamol inhaler",
            "chlorpheniramine": "Chlorpheniramine",
            "antifungal": "Topical antifungal",
            "antibiotic ointment": "Antibiotic ointment",
        }

        for line in response.split("\n"):
            line_lower = line.lower().strip()
            for med_key, med_name in known_meds.items():
                if med_key in line_lower and med_name not in found_meds:
                    dosage = self._extract_dosage_from_line(line, med_key)
                    medications.append(
                        Medication(name=med_name, dosage=dosage),
                    )
                    found_meds.add(med_name)
                    break

        if not medications and "supportive" in response.lower():
            medications.append(
                Medication(
                    name="Supportive care",
                    dosage="Rest, fluids, monitor symptoms",
                ),
            )

        return medications

    def _extract_dosage_from_line(
        self, line: str, med_keyword: str,
    ) -> str:
        """Extract dosage from a line containing a medication name.

        Args:
            line: Source line of text.
            med_keyword: Keyword that identified the medication.

        Returns:
            Dosage string, or a sensible default.
        """
        line_lower = line.lower()
        med_pos = line_lower.find(med_keyword)
        if med_pos == -1:
            return "as directed"

        after_med = line[med_pos + len(med_keyword):].strip()
        after_med = re.sub(r"^[\s:\-\*\u2022]+", "", after_med).strip()

        dosage_patterns = [
            r"(\d+\s*(?:mg|ml|g|mcg|iu|units?)[^.]*"
            r"(?:daily|hourly|hours|times|twice|once|per day|every)[^.]*)",
            r"(\d+\s*tablets?[^.]*(?:daily|times|twice|once)[^.]*)",
            r"((?:apply|take|give|use)\s+[^.]{5,40})",
            r"(\d+\s*(?:mg|ml)[^.]*for\s+\d+\s*days[^.]*)",
            r"(\d+\s*(?:mg|ml|g|tablets?|capsules?|puffs?)[^,.\n]{0,30})",
        ]

        for pattern in dosage_patterns:
            match = re.search(pattern, after_med, re.IGNORECASE)
            if match:
                dosage = re.sub(r"\s+", " ", match.group(1).strip())
                if len(dosage) > 5:
                    return dosage

        for pattern in dosage_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                dosage = re.sub(r"\s+", " ", match.group(1).strip())
                if len(dosage) > 5:
                    return dosage

        defaults: Dict[str, str] = {
            "paracetamol": "500-1000mg every 4-6 hours (max 4g/day)",
            "acetaminophen": "500-1000mg every 4-6 hours (max 4g/day)",
            "amoxicillin": "500mg every 8 hours for 5-7 days",
            "ibuprofen": "400mg every 6-8 hours with food",
            "oral rehydration": "As needed for hydration",
            "ors": "As needed for hydration",
            "zinc": "20mg daily for 10-14 days",
            "vitamin a": "200,000 IU single dose",
            "artemether": "4 tablets at 0, 8, 24, 36, 48, 60 hours",
            "metronidazole": "400mg every 8 hours for 5-7 days",
            "clotrimazole": "Apply twice daily for 2-4 weeks",
            "hydrocortisone": "Apply thin layer 1-2 times daily",
        }
        return defaults.get(med_keyword, "as directed")

    def _extract_instructions(self, response: str) -> List[str]:
        """Extract non-medication instructions from a response.

        Args:
            response: Raw model response text.

        Returns:
            Up to 5 instruction strings.
        """
        instructions: List[str] = []
        instruction_keywords = [
            "rest", "fluid", "drink", "hydrat", "clean", "wash",
            "monitor", "observe", "return", "avoid", "keep",
        ]

        for line in response.split("\n"):
            line_lower = line.lower().strip()
            if any(kw in line_lower for kw in instruction_keywords):
                line = re.sub(r"^[\-\*\d\.]+\s*", "", line).strip()
                if line and len(line) > 10:
                    instructions.append(line)

        if not instructions:
            instructions = [
                "Ensure adequate rest",
                "Maintain hydration with clean water",
                "Return if symptoms worsen",
            ]
        return instructions[:5]

    def _extract_warning_signs(self, response: str) -> List[str]:
        """Extract warning signs from a response.

        Args:
            response: Raw model response text.

        Returns:
            Up to 5 warning sign strings.
        """
        warning_signs: List[str] = []

        warning_section = re.search(
            r"warning[s]?[:\s]*(.*?)(?=\n\n|follow|referral|$)",
            response.lower(), re.DOTALL,
        )
        if warning_section:
            items = re.split(r"[\n\-\*]+", warning_section.group(1))
            for item in items:
                item = item.strip()
                if item and len(item) > 5:
                    warning_signs.append(item)

        if not warning_signs:
            warning_signs = [
                "High fever that doesn't respond to paracetamol",
                "Difficulty breathing",
                "Unable to drink or eat",
                "Altered consciousness or confusion",
                "Severe vomiting or diarrhea",
            ][:3]

        return warning_signs[:5]

    def _extract_referral(
        self, response: str,
    ) -> Tuple[bool, Optional[str]]:
        """Determine if referral is recommended from a response.

        Args:
            response: Raw model response text.

        Returns:
            Tuple of ``(needs_referral, reason_or_none)``.
        """
        response_lower = response.lower()

        no_referral_phrases = [
            "no referral", "can be managed", "does not require",
            "outpatient", "health post level",
        ]
        for phrase in no_referral_phrases:
            if phrase in response_lower:
                return False, None

        referral_phrases = [
            "refer", "hospital", "emergency", "urgent",
            "cannot be managed", "beyond scope",
        ]
        for phrase in referral_phrases:
            if phrase in response_lower:
                match = re.search(
                    rf"{phrase}[:\s]*([^\n\.]+)", response_lower,
                )
                reason = (
                    match.group(1).strip()
                    if match
                    else "Requires higher level care"
                )
                return True, reason

        return False, None

    def _determine_follow_up(self, condition: str) -> int:
        """Determine follow-up interval based on condition type.

        Args:
            condition: Diagnosis condition name.

        Returns:
            Recommended follow-up in days.
        """
        condition_lower = condition.lower()

        for keyword in ("fever", "infection", "malaria", "diarrhea"):
            if keyword in condition_lower:
                return 2

        for keyword in ("skin", "wound", "rash"):
            if keyword in condition_lower:
                return 5

        return 3
