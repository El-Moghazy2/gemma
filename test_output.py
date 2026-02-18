"""Test the output format with drug interaction section and alternatives."""

from healthpost.core import PatientVisitResult
from healthpost.triage import Diagnosis, TreatmentPlan, Medication
from healthpost.drugs import DrugInteraction

# Create a sample result with interactions
diagnosis = Diagnosis(
    condition="Conjunctivitis",
    confidence=0.7,
    supporting_evidence=[
        "Symptoms: eye feels itchy",
        "Red and inflamed eye visible in image",
    ],
)

treatment = TreatmentPlan(
    medications=[
        Medication(name="Erythromycin eye drops", dosage="1 drop 3 times daily for 5-7 days"),
        Medication(name="Artificial tears", dosage="as needed for comfort"),
    ],
    instructions=[
        "Wash hands frequently with soap and water",
        "Avoid touching the affected eye(s)",
        "Do not share towels or washcloths",
    ],
    follow_up_days=3,
)

# Create sample interactions (for demo)
interactions = [
    DrugInteraction(
        drugs=("Erythromycin", "Warfarin"),
        severity="moderate",
        description="Erythromycin may increase the anticoagulant effect of warfarin, increasing bleeding risk.",
        recommendation="Monitor INR closely when starting or stopping erythromycin.",
    )
]

# Create sample alternatives
alternatives = {
    "Erythromycin eye drops": "Tobramycin eye drops: 1 drop every 4 hours for 7 days"
}

# Create result WITH interactions and alternatives
result_with_interactions = PatientVisitResult(
    symptoms_text="eye feels itchy",
    visual_findings=["Red and inflamed eye"],
    current_medications=["Warfarin"],
    diagnosis=diagnosis,
    treatment_plan=treatment,
    drug_interactions=interactions,
    is_safe_to_proceed=True,  # moderate, not severe
    alternative_medications=alternatives,
    needs_referral=False,
    referral_reason=None,
    overall_confidence=0.7,
)

# Create result WITHOUT interactions
result_no_interactions = PatientVisitResult(
    symptoms_text="eye feels itchy",
    visual_findings=["Red and inflamed eye"],
    current_medications=[],
    diagnosis=diagnosis,
    treatment_plan=treatment,
    drug_interactions=[],
    is_safe_to_proceed=True,
    alternative_medications={},
    needs_referral=False,
    referral_reason=None,
    overall_confidence=0.7,
)

# Write to file to avoid encoding issues
with open("output_sample.txt", "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("EXAMPLE 1: With current medications, interactions, and alternatives\n")
    f.write("=" * 60 + "\n")
    f.write(result_with_interactions.format_for_display())
    f.write("\n\n\n")
    f.write("=" * 60 + "\n")
    f.write("EXAMPLE 2: No current medications, no interactions\n")
    f.write("=" * 60 + "\n")
    f.write(result_no_interactions.format_for_display())

print("Output written to output_sample.txt")
