"""
Core HealthPost orchestrator - manages the complete patient visit workflow.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging

from .config import Config, default_config
from .voice import VoiceTranscriber
from .vision import MedicalVisionAnalyzer
from .drugs import DrugDatabase, DrugInteraction
from .triage import TriageAgent, Diagnosis, TreatmentPlan

logger = logging.getLogger(__name__)


@dataclass
class PatientVisitResult:
    """Complete result from a patient visit workflow."""

    # Input summaries
    symptoms_text: str
    visual_findings: List[str]
    current_medications: List[str]

    # Diagnosis
    diagnosis: Diagnosis

    # Treatment plan
    treatment_plan: TreatmentPlan

    # Safety checks
    drug_interactions: List[DrugInteraction]
    is_safe_to_proceed: bool

    # Referral
    needs_referral: bool
    referral_reason: Optional[str]

    # Confidence
    overall_confidence: float

    # Alternative medications (if interactions found) - has default so comes last
    alternative_medications: Dict[str, str] = field(default_factory=dict)  # {problematic_drug: alternative}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "symptoms": self.symptoms_text,
            "visual_findings": self.visual_findings,
            "current_medications": self.current_medications,
            "diagnosis": {
                "condition": self.diagnosis.condition,
                "confidence": self.diagnosis.confidence,
                "supporting_evidence": self.diagnosis.supporting_evidence,
                "differential": self.diagnosis.differential_diagnoses,
            },
            "treatment": {
                "medications": [
                    {"name": m.name, "dosage": m.dosage, "duration": m.duration}
                    for m in self.treatment_plan.medications
                ],
                "instructions": self.treatment_plan.instructions,
                "follow_up": self.treatment_plan.follow_up_days,
            },
            "safety": {
                "interactions": [
                    {"drugs": i.drugs, "severity": i.severity, "description": i.description}
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
        """Format result for CHW display."""
        lines = []

        # Header
        lines.append("=" * 50)
        lines.append("HEALTHPOST - PATIENT VISIT SUMMARY")
        lines.append("=" * 50)

        # Diagnosis
        lines.append(f"\n📋 DIAGNOSIS: {self.diagnosis.condition}")
        lines.append(f"   Confidence: {self.diagnosis.confidence:.0%}")

        if self.diagnosis.supporting_evidence:
            lines.append("   Evidence:")
            for ev in self.diagnosis.supporting_evidence[:3]:
                lines.append(f"   • {ev}")

        # Treatment
        lines.append("\n💊 RECOMMENDED TREATMENT:")
        for med in self.treatment_plan.medications:
            lines.append(f"   • {med.name}: {med.dosage}")
            if med.duration:
                lines.append(f"     Duration: {med.duration}")

        if self.treatment_plan.instructions:
            lines.append("\n📝 INSTRUCTIONS:")
            for instr in self.treatment_plan.instructions:
                lines.append(f"   • {instr}")

        # Drug Interactions Section
        lines.append("\n💊 DRUG INTERACTION CHECK:")

        # Show what was checked
        all_meds = self.current_medications + [m.name for m in self.treatment_plan.medications]
        if all_meds:
            lines.append(f"   Medications checked: {', '.join(all_meds)}")
        else:
            lines.append("   No medications to check")

        if self.drug_interactions:
            lines.append(f"\n   ⚠️ {len(self.drug_interactions)} INTERACTION(S) FOUND:")
            for interaction in self.drug_interactions:
                if interaction.severity == "severe":
                    severity_icon = "🔴 SEVERE"
                elif interaction.severity == "moderate":
                    severity_icon = "🟡 MODERATE"
                else:
                    severity_icon = "🟢 MILD"
                lines.append(f"   {severity_icon}: {' + '.join(interaction.drugs)}")
                lines.append(f"      {interaction.description[:150]}...")
                if interaction.recommendation:
                    lines.append(f"      Recommendation: {interaction.recommendation[:100]}...")

            # Show alternative medications if available
            if self.alternative_medications:
                lines.append("\n   💡 SUGGESTED ALTERNATIVES:")
                for drug, alternative in self.alternative_medications.items():
                    lines.append(f"      Instead of {drug}: {alternative}")
        else:
            lines.append("\n   ✓ No interactions detected")

        # Safety summary
        if self.is_safe_to_proceed:
            lines.append("\n✅ SAFE TO PROCEED with treatment")
        else:
            lines.append("\n❌ DO NOT PROCEED - Review interactions above")

        # Referral
        if self.needs_referral:
            lines.append(f"\n🏥 REFERRAL NEEDED: {self.referral_reason}")

        # Follow-up
        if self.treatment_plan.follow_up_days:
            lines.append(f"\n📅 Follow-up in {self.treatment_plan.follow_up_days} days")

        lines.append("\n" + "=" * 50)

        return "\n".join(lines)


class HealthPost:
    """
    Complete CHW Decision Support System.

    Orchestrates the full patient visit workflow:
    1. INTAKE - Voice transcription of symptoms
    2. DIAGNOSE - Image analysis + reasoning
    3. PRESCRIBE - Treatment recommendations
    4. DISPENSE - Drug interaction safety check
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize HealthPost with configuration."""
        self.config = config or default_config

        # Components (lazy loaded)
        self._voice: Optional[VoiceTranscriber] = None
        self._vision: Optional[MedicalVisionAnalyzer] = None
        self._drug_db: Optional[DrugDatabase] = None
        self._triage: Optional[TriageAgent] = None

        self._initialized = False

        logger.info("HealthPost instance created")

    def initialize(self):
        """Initialize all components. Call before first use."""
        if self._initialized:
            return

        logger.info("Initializing HealthPost components...")

        # Initialize drug database first (lightweight)
        self._drug_db = DrugDatabase(self.config.drug_db_path)
        logger.info("Drug database loaded")

        # Initialize AI models
        self._voice = VoiceTranscriber(self.config)
        logger.info("Voice transcriber initialized")

        self._vision = MedicalVisionAnalyzer(self.config)
        logger.info("Vision analyzer initialized")

        self._triage = TriageAgent(self.config, self._vision)
        logger.info("Triage agent initialized")

        self._initialized = True
        logger.info("HealthPost fully initialized")

    @property
    def voice(self) -> VoiceTranscriber:
        """Get voice transcriber component."""
        if self._voice is None:
            self.initialize()
        return self._voice

    @property
    def vision(self) -> MedicalVisionAnalyzer:
        """Get vision analyzer component."""
        if self._vision is None:
            self.initialize()
        return self._vision

    @property
    def drug_db(self) -> DrugDatabase:
        """Get drug database component."""
        if self._drug_db is None:
            self.initialize()
        return self._drug_db

    @property
    def triage(self) -> TriageAgent:
        """Get triage agent component."""
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
    ) -> PatientVisitResult:
        """
        Complete patient visit workflow.

        Args:
            audio: Audio recording of symptom description (numpy array or file path)
            symptoms_text: Text description of symptoms (alternative to audio)
            images: List of medical images (skin, wound, eye photos)
            existing_meds_photo: Photo of patient's current medications
            existing_meds_list: List of current medication names (alternative to photo)

        Returns:
            PatientVisitResult with diagnosis, treatment, and safety information
        """
        self.initialize()

        # Step 1: INTAKE - Capture symptoms
        if audio is not None:
            symptoms = self.voice.transcribe(audio)
        elif symptoms_text:
            symptoms = symptoms_text
        else:
            symptoms = "No symptoms provided"

        logger.info(f"Symptoms captured: {symptoms[:100]}...")

        # Step 2: DIAGNOSE - Analyze images
        visual_findings = []
        if images:
            for img in images:
                findings = self.vision.analyze_medical_image(img)
                visual_findings.extend(findings)

        logger.info(f"Visual findings: {len(visual_findings)} items")

        # Step 3: DIAGNOSE + PRESCRIBE - Generate diagnosis and treatment
        diagnosis, treatment = self.triage.diagnose_and_treat(
            symptoms=symptoms,
            visual_findings=visual_findings,
        )

        logger.info(f"Diagnosis: {diagnosis.condition} ({diagnosis.confidence:.0%})")

        # Step 4: DISPENSE - Extract current medications
        current_meds = []
        if existing_meds_photo is not None:
            current_meds = self.vision.extract_medications(existing_meds_photo)
        if existing_meds_list:
            current_meds.extend(existing_meds_list)
        current_meds = list(set(current_meds))  # Deduplicate

        logger.info(f"Current medications: {current_meds}")

        # Step 5: Safety check - Drug interactions
        proposed_meds = [m.name for m in treatment.medications]
        all_meds = current_meds + proposed_meds
        interactions = self.drug_db.check_interactions(all_meds)

        # Determine if safe to proceed
        severe_interactions = [i for i in interactions if i.severity == "severe"]
        is_safe = len(severe_interactions) == 0

        logger.info(f"Interactions found: {len(interactions)}, Safe: {is_safe}")

        # Step 5b: Get alternative medications if interactions found
        alternative_medications = {}
        if interactions:
            alternative_medications = self._get_alternative_medications(
                diagnosis, treatment, interactions, current_meds
            )

        # Step 6: Determine if referral needed
        needs_referral, referral_reason = self._check_referral_needed(
            diagnosis, treatment, interactions
        )

        # Calculate overall confidence
        overall_confidence = self._calculate_confidence(
            diagnosis, visual_findings, interactions
        )

        return PatientVisitResult(
            symptoms_text=symptoms,
            visual_findings=visual_findings,
            current_medications=current_meds,
            diagnosis=diagnosis,
            treatment_plan=treatment,
            drug_interactions=interactions,
            is_safe_to_proceed=is_safe,
            alternative_medications=alternative_medications,
            needs_referral=needs_referral,
            referral_reason=referral_reason,
            overall_confidence=overall_confidence,
        )

    def _check_referral_needed(
        self,
        diagnosis: Diagnosis,
        treatment: TreatmentPlan,
        interactions: List[DrugInteraction],
    ) -> tuple[bool, Optional[str]]:
        """Determine if patient should be referred to hospital."""

        # Low confidence diagnosis
        if diagnosis.confidence < self.config.confidence_threshold:
            return True, f"Low diagnostic confidence ({diagnosis.confidence:.0%})"

        # Severe drug interactions
        severe = [i for i in interactions if i.severity == "severe"]
        if severe:
            return True, f"Severe drug interaction detected: {severe[0].drugs}"

        # Conditions requiring referral (emergencies)
        emergency_conditions = [
            "sepsis", "meningitis", "severe pneumonia", "heart attack",
            "stroke", "severe dehydration", "severe malaria",
        ]
        for cond in emergency_conditions:
            if cond.lower() in diagnosis.condition.lower():
                return True, f"Emergency condition: {diagnosis.condition}"

        # Treatment requires referral
        if treatment.requires_referral:
            return True, treatment.referral_reason

        return False, None

    def _calculate_confidence(
        self,
        diagnosis: Diagnosis,
        visual_findings: List[str],
        interactions: List[DrugInteraction],
    ) -> float:
        """Calculate overall confidence score."""
        confidence = diagnosis.confidence

        # Boost if visual findings support diagnosis
        if visual_findings:
            confidence = min(1.0, confidence + 0.1)

        # Reduce if interactions found
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
        """Get alternative medications for those with interactions."""
        alternatives = {}

        # Find which recommended drugs are causing interactions
        recommended_drugs = {m.name.lower(): m.name for m in treatment.medications}

        for interaction in interactions:
            for drug in interaction.drugs:
                drug_lower = drug.lower()
                # Check if this drug is one of our recommendations (not patient's current meds)
                for rec_key, rec_name in recommended_drugs.items():
                    if drug_lower in rec_key or rec_key in drug_lower:
                        if rec_name not in alternatives:
                            # Ask triage agent for alternative
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
        """Ask the triage agent to suggest an alternative medication."""
        prompt = f"""A patient with {condition} needs treatment.

The recommended medication "{problematic_drug}" has an interaction with their current medication(s).

Interaction details:
- Drugs involved: {', '.join(interaction.drugs)}
- Severity: {interaction.severity}
- Description: {interaction.description}

Patient's current medications: {', '.join(current_meds) if current_meds else 'None specified'}

Suggest ONE alternative medication that:
1. Can treat {condition}
2. Does not interact with the patient's current medications
3. Is commonly available

Respond with ONLY the alternative medication name and dosage, nothing else.
Example: "Tobramycin eye drops: 1 drop every 4 hours"
"""
        try:
            response = self.triage._generate_response(prompt)
            # Clean up response
            alternative = response.strip()
            # Remove common prefixes
            for prefix in ["Alternative:", "Suggestion:", "Instead use:", "Use:"]:
                if alternative.lower().startswith(prefix.lower()):
                    alternative = alternative[len(prefix):].strip()
            # Limit length
            if len(alternative) > 100:
                alternative = alternative[:100]
            return alternative if alternative else None
        except Exception as e:
            logger.warning(f"Failed to get alternative medication: {e}")
            return None

    # Convenience methods for individual steps

    def transcribe_symptoms(self, audio: Any) -> str:
        """Transcribe audio recording of symptoms."""
        return self.voice.transcribe(audio)

    def analyze_image(self, image: Any) -> List[str]:
        """Analyze a medical image."""
        return self.vision.analyze_medical_image(image)

    def extract_medications(self, image: Any) -> List[str]:
        """Extract medication names from a photo."""
        return self.vision.extract_medications(image)

    def check_drug_interactions(self, medications: List[str]) -> List[DrugInteraction]:
        """Check for drug interactions."""
        return self.drug_db.check_interactions(medications)
