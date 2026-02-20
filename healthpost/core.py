"""Core HealthPost orchestrator managing the complete patient visit workflow."""

import logging
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import Config, default_config
from .drugs import DrugDatabase, DrugInteraction
from .inference_backend import create_backend
from .triage import Diagnosis, Medication, TreatmentPlan, TriageAgent
from .vision import MedicalVisionAnalyzer
from .voice import VoiceTranscriber

logger = logging.getLogger(__name__)


class PatientVisitResult(BaseModel):
    """Complete result from a patient visit workflow."""

    symptoms_text: str
    visual_findings: List[str]
    current_medications: List[str]
    diagnosis: Diagnosis
    treatment_plan: TreatmentPlan
    drug_interactions: List[DrugInteraction]
    is_safe_to_proceed: bool
    needs_referral: bool
    referral_reason: Optional[str]
    overall_confidence: float
    alternative_medications: Dict[str, str] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the result to a plain dictionary."""
        return {
            "symptoms": self.symptoms_text,
            "visual_findings": self.visual_findings,
            "current_medications": self.current_medications,
            "diagnosis": {
                "condition": self.diagnosis.condition,
                "confidence": self.diagnosis.confidence,
                "supporting_evidence": self.diagnosis.supporting_evidence,
                "differential": self.diagnosis.differential_diagnoses,
                "known_symptoms": self.diagnosis.known_symptoms,
            },
            "treatment": {
                "medications": [
                    {
                        "name": m.name,
                        "dosage": m.dosage,
                        "duration": m.duration,
                    }
                    for m in self.treatment_plan.medications
                ],
                "instructions": self.treatment_plan.instructions,
                "follow_up": self.treatment_plan.follow_up_days,
            },
            "safety": {
                "interactions": [
                    {
                        "drugs": i.drugs,
                        "severity": i.severity,
                        "description": i.description,
                    }
                    for i in self.drug_interactions
                ],
                "safe_to_proceed": self.is_safe_to_proceed,
            },
            "referral": {
                "needed": self.needs_referral,
                "reason": self.referral_reason,
            },
            "confidence": self.overall_confidence,
        }

    def format_for_display(self) -> str:
        """Format the result as a plain-text summary for CHW display."""
        lines: List[str] = []

        lines.append("=" * 50)
        lines.append("HEALTHPOST - PATIENT VISIT SUMMARY")
        lines.append("=" * 50)

        lines.append(f"\nDIAGNOSIS: {self.diagnosis.condition}")
        lines.append(f"   Confidence: {self.diagnosis.confidence:.0%}")

        if self.diagnosis.supporting_evidence:
            lines.append("   Evidence:")
            for ev in self.diagnosis.supporting_evidence[:3]:
                lines.append(f"   - {ev}")

        if self.diagnosis.known_symptoms:
            lines.append("\nKNOWN SYMPTOMS of this condition:")
            for sym in self.diagnosis.known_symptoms:
                lines.append(f"   - {sym}")

        lines.append("\nRECOMMENDED TREATMENT:")
        for med in self.treatment_plan.medications:
            lines.append(f"   - {med.name}: {med.dosage}")
            if med.duration:
                lines.append(f"     Duration: {med.duration}")

        if self.treatment_plan.instructions:
            lines.append("\nINSTRUCTIONS:")
            for instr in self.treatment_plan.instructions:
                lines.append(f"   - {instr}")

        lines.append("\nDRUG INTERACTION CHECK:")
        all_meds = self.current_medications + [
            m.name for m in self.treatment_plan.medications
        ]
        if all_meds:
            lines.append(
                f"   Medications checked: {', '.join(all_meds)}"
            )
        else:
            lines.append("   No medications to check")

        if self.drug_interactions:
            lines.append(
                f"\n   {len(self.drug_interactions)} INTERACTION(S) "
                "FOUND:"
            )
            for interaction in self.drug_interactions:
                severity_label = interaction.severity.upper()
                drug_pair = " + ".join(interaction.drugs)
                lines.append(f"   {severity_label}: {drug_pair}")
                lines.append(
                    f"      {interaction.description[:150]}..."
                )
                if interaction.recommendation:
                    lines.append(
                        "      Recommendation: "
                        f"{interaction.recommendation[:100]}..."
                    )

            if self.alternative_medications:
                lines.append("\n   SUGGESTED ALTERNATIVES:")
                for drug, alternative in (
                    self.alternative_medications.items()
                ):
                    lines.append(
                        f"      Instead of {drug}: {alternative}"
                    )
        else:
            lines.append("\n   No interactions detected")

        if self.is_safe_to_proceed:
            lines.append("\nSAFE TO PROCEED with treatment")
        else:
            lines.append(
                "\nDO NOT PROCEED - Review interactions above"
            )

        if self.needs_referral:
            lines.append(f"\nREFERRAL NEEDED: {self.referral_reason}")

        if self.treatment_plan.follow_up_days:
            lines.append(
                f"\nFollow-up in {self.treatment_plan.follow_up_days} "
                "days"
            )

        lines.append("\n" + "=" * 50)
        return "\n".join(lines)


class HealthPost:
    """Complete CHW Decision Support System.

    Orchestrates the full patient visit workflow:

    1. **INTAKE** -- Voice transcription of symptoms.
    2. **DIAGNOSE** -- Image analysis and reasoning.
    3. **PRESCRIBE** -- Treatment recommendations.
    4. **DISPENSE** -- Drug interaction safety check.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or default_config

        self._voice: Optional[VoiceTranscriber] = None
        self._vision: Optional[MedicalVisionAnalyzer] = None
        self._drug_db: Optional[DrugDatabase] = None
        self._triage: Optional[TriageAgent] = None
        self._visit_graph = None
        self._initialized = False

        logger.info("HealthPost instance created")

    def initialize(self) -> None:
        """Eagerly initialize all subsystem components."""
        if self._initialized:
            return

        logger.info("Initializing HealthPost components...")

        self._drug_db = DrugDatabase()
        logger.info("Drug database loaded")

        self._voice = VoiceTranscriber(self.config)
        logger.info("Voice transcriber initialized")

        backend = create_backend(self.config)
        logger.info("Inference backend created: %s", type(backend).__name__)

        self._vision = MedicalVisionAnalyzer(self.config, backend=backend)
        logger.info("Vision analyzer initialized")

        self._triage = TriageAgent(self.config, self._vision, backend=backend)
        logger.info("Triage agent initialized")

        from .visit_graph import build_visit_graph
        self._visit_graph = build_visit_graph(self)
        logger.info("Visit graph compiled")

        self._initialized = True
        logger.info("HealthPost fully initialized")

    @property
    def voice(self) -> VoiceTranscriber:
        """Lazily initialized voice transcriber."""
        if self._voice is None:
            self.initialize()
        return self._voice

    @property
    def vision(self) -> MedicalVisionAnalyzer:
        """Lazily initialized vision analyzer."""
        if self._vision is None:
            self.initialize()
        return self._vision

    @property
    def drug_db(self) -> DrugDatabase:
        """Lazily initialized drug database."""
        if self._drug_db is None:
            self.initialize()
        return self._drug_db

    @property
    def triage(self) -> TriageAgent:
        """Lazily initialized triage agent."""
        if self._triage is None:
            self.initialize()
        return self._triage

    def patient_visit(
        self,
        audio: Optional[Any] = None,
        symptoms_text: Optional[str] = None,
        images: Optional[List[Any]] = None,
        existing_meds_photo: Optional[Any] = None,
        existing_meds_list: Optional[List[str]] = None,
        patient_age: Optional[str] = None,
        on_progress: Optional[Callable[[str, str], None]] = None,
    ) -> PatientVisitResult:
        """Run the patient visit workflow.

        Args:
            audio: Audio recording of symptoms.
            symptoms_text: Text description of symptoms.
            images: Medical images for analysis.
            existing_meds_photo: Photo of current medications.
            existing_meds_list: List of current medication names.
            patient_age: Patient age description.
            on_progress: Optional callback ``(step_name, description)``
                for pipeline progress updates.

        Returns:
            ``PatientVisitResult`` with diagnosis, treatment, and
            safety information.
        """
        self.initialize()

        visit_state = {
            "audio": audio,
            "symptoms_text": symptoms_text,
            "images": images,
            "existing_meds_photo": existing_meds_photo,
            "existing_meds_list": existing_meds_list,
            "patient_age": patient_age,
        }

        if on_progress:
            from .visit_graph import build_visit_graph
            graph = build_visit_graph(self, on_progress=on_progress)
            final_state = graph.invoke(visit_state)
        else:
            final_state = self._visit_graph.invoke(visit_state)
        return self._state_to_result(final_state)

    def _state_to_result(self, state: Dict[str, Any]) -> PatientVisitResult:
        """Convert a final visit graph state to a ``PatientVisitResult``."""
        return PatientVisitResult(
            symptoms_text=state.get("symptoms", ""),
            visual_findings=state.get("visual_findings", []),
            current_medications=state.get("current_meds", []),
            diagnosis=state.get("diagnosis"),
            treatment_plan=state.get("treatment_plan"),
            drug_interactions=state.get("drug_interactions", []),
            is_safe_to_proceed=state.get("is_safe_to_proceed", True),
            alternative_medications=state.get(
                "alternative_medications", {},
            ),
            needs_referral=state.get("needs_referral", False),
            referral_reason=state.get("referral_reason"),
            overall_confidence=state.get("overall_confidence", 0.7),
        )

    def _check_referral_needed(
        self,
        diagnosis: Diagnosis,
        treatment: TreatmentPlan,
        interactions: List[DrugInteraction],
    ) -> tuple[bool, Optional[str]]:
        """Determine whether the patient should be referred."""
        if diagnosis.confidence < self.config.confidence_threshold:
            return (
                True,
                f"Low diagnostic confidence "
                f"({diagnosis.confidence:.0%})",
            )

        severe = [i for i in interactions if i.severity == "severe"]
        if severe:
            return (
                True,
                f"Severe drug interaction detected: {severe[0].drugs}",
            )

        emergency_conditions = [
            "sepsis", "meningitis", "severe pneumonia",
            "heart attack", "stroke", "severe dehydration",
            "severe malaria",
        ]
        for cond in emergency_conditions:
            if cond.lower() in diagnosis.condition.lower():
                return (
                    True,
                    f"Emergency condition: {diagnosis.condition}",
                )

        if treatment.requires_referral:
            return True, treatment.referral_reason

        return False, None

    def _calculate_confidence(
        self,
        diagnosis: Diagnosis,
        visual_findings: List[str],
        interactions: List[DrugInteraction],
    ) -> float:
        """Calculate an aggregate confidence score."""
        confidence = diagnosis.confidence

        if visual_findings:
            confidence = min(1.0, confidence + 0.1)
        if interactions:
            confidence = max(0.0, confidence - 0.1 * len(interactions))

        return confidence

    def _get_alternative_medications(
        self,
        diagnosis: Diagnosis,
        treatment: TreatmentPlan,
        interactions: List[DrugInteraction],
        current_meds: List[str],
    ) -> Dict[str, str]:
        """Suggest alternatives for recommended drugs with interactions."""
        alternatives: Dict[str, str] = {}
        recommended_drugs = {
            m.name.lower(): m.name for m in treatment.medications
        }

        for interaction in interactions:
            for drug in interaction.drugs:
                drug_lower = drug.lower()
                for rec_key, rec_name in recommended_drugs.items():
                    if drug_lower in rec_key or rec_key in drug_lower:
                        if rec_name not in alternatives:
                            alternative = self._suggest_alternative(
                                diagnosis.condition,
                                rec_name,
                                interaction,
                                current_meds,
                            )
                            if alternative:
                                alternatives[rec_name] = alternative

        return alternatives

    def _suggest_alternative(
        self,
        condition: str,
        problematic_drug: str,
        interaction: DrugInteraction,
        current_meds: List[str],
    ) -> Optional[str]:
        """Ask the triage agent to suggest a single alternative drug."""
        current_str = (
            ", ".join(current_meds)
            if current_meds
            else "None specified"
        )
        prompt = (
            f'A patient with {condition} needs treatment.\n\n'
            f'The recommended medication "{problematic_drug}" has an '
            f'interaction with their current medication(s).\n\n'
            f'Interaction details:\n'
            f'- Drugs involved: {", ".join(interaction.drugs)}\n'
            f'- Severity: {interaction.severity}\n'
            f'- Description: {interaction.description}\n\n'
            f"Patient's current medications: {current_str}\n\n"
            f'Suggest ONE alternative medication that:\n'
            f'1. Can treat {condition}\n'
            f'2. Does not interact with the patient\'s current '
            f'medications\n'
            f'3. Is commonly available\n\n'
            f'Respond with ONLY the alternative medication name and '
            f'dosage, nothing else.\n'
            f'Example: "Tobramycin eye drops: 1 drop every 4 hours"'
        )

        try:
            response = self.triage._generate_response(prompt)
            alternative = response.strip()
            for prefix in (
                "Alternative:", "Suggestion:", "Instead use:", "Use:",
            ):
                if alternative.lower().startswith(prefix.lower()):
                    alternative = alternative[len(prefix):].strip()
            if len(alternative) > 100:
                alternative = alternative[:100]
            return alternative if alternative else None
        except Exception as e:
            logger.warning(
                "Failed to get alternative medication: %s", e,
            )
            return None

    def transcribe_symptoms(self, audio: Any) -> str:
        """Transcribe an audio recording of symptoms."""
        return self.voice.transcribe(audio)

    def analyze_image(self, image: Any) -> List[str]:
        """Analyze a medical image."""
        return self.vision.analyze_medical_image(image)

    def extract_medications(self, image: Any) -> List[str]:
        """Extract medication names from a photo."""
        return self.vision.extract_medications(image)

    def check_drug_interactions(
        self, medications: List[str],
    ) -> List[DrugInteraction]:
        """Check for drug-drug interactions."""
        return self.drug_db.check_interactions(medications)
