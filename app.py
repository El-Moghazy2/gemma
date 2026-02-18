"""
HealthPost - Complete CHW Decision Support System

Gradio web interface for Community Health Workers.
Supports the complete patient visit workflow:
1. INTAKE - Voice/text symptom capture
2. DIAGNOSE - Image analysis
3. PRESCRIBE - AI-generated treatment
4. DISPENSE - Drug safety check
"""

import gradio as gr
from typing import Optional, List, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import HealthPost components
from healthpost import HealthPost, Config

# Global instance (lazy loaded)
_healthpost: Optional[HealthPost] = None


def get_healthpost() -> HealthPost:
    """Get or create the HealthPost instance."""
    global _healthpost
    if _healthpost is None:
        logger.info("Initializing HealthPost...")
        config = Config()
        _healthpost = HealthPost(config)
    return _healthpost


# ============================================================================
# STEP 1: INTAKE - Symptom Capture
# ============================================================================

def transcribe_audio(audio: Any) -> str:
    """Transcribe audio recording of symptoms."""
    if audio is None:
        return ""

    try:
        hp = get_healthpost()
        text = hp.transcribe_symptoms(audio)
        return text
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return f"[Error transcribing audio: {e}]"


# ============================================================================
# STEP 2: DIAGNOSE - Image Analysis
# ============================================================================

def analyze_medical_image(image: Any, image_type: str) -> str:
    """Analyze a medical image."""
    if image is None:
        return ""

    try:
        hp = get_healthpost()

        if image_type == "Skin/Rash":
            result = hp.vision.analyze_skin_condition(image)
        elif image_type == "Wound":
            result = hp.vision.analyze_wound(image)
        else:
            findings = hp.vision.analyze_medical_image(image)
            result = {"findings": findings}

        # Format output
        if "raw_analysis" in result:
            return result["raw_analysis"]
        elif "findings" in result:
            return "\n".join(f"• {f}" for f in result["findings"])
        else:
            return str(result)

    except Exception as e:
        logger.error(f"Image analysis error: {e}")
        return f"[Error analyzing image: {e}]"


# ============================================================================
# STEP 3: PRESCRIBE - Diagnosis & Treatment
# ============================================================================

def generate_diagnosis(
    symptoms_text: str,
    visual_findings: str,
    patient_age: str,
) -> Tuple[str, str, str]:
    """Generate diagnosis and treatment plan."""
    if not symptoms_text.strip():
        return "Please provide symptoms", "", ""

    try:
        hp = get_healthpost()

        # Parse visual findings
        findings_list = []
        if visual_findings.strip():
            for line in visual_findings.split('\n'):
                line = line.strip().lstrip('•-* ')
                if line:
                    findings_list.append(line)

        # Generate diagnosis
        diagnosis, treatment = hp.triage.diagnose_and_treat(
            symptoms=symptoms_text,
            visual_findings=findings_list,
            patient_age=patient_age if patient_age else None,
        )

        # Format diagnosis output
        diagnosis_text = f"""**{diagnosis.condition}**

Confidence: {diagnosis.confidence:.0%}

**Supporting Evidence:**
{chr(10).join(f'• {e}' for e in diagnosis.supporting_evidence)}

**Consider Also:**
{chr(10).join(f'• {d}' for d in diagnosis.differential_diagnoses) if diagnosis.differential_diagnoses else '• None'}
"""

        # Format treatment output
        meds_text = "\n".join(
            f"• **{m.name}**: {m.dosage}" + (f" for {m.duration}" if m.duration else "")
            for m in treatment.medications
        ) if treatment.medications else "• Supportive care only"

        treatment_text = f"""**Medications:**
{meds_text}

**Instructions:**
{chr(10).join(f'• {i}' for i in treatment.instructions)}

**Warning Signs (Return if):**
{chr(10).join(f'• {w}' for w in treatment.warning_signs)}

**Follow-up:** {treatment.follow_up_days} days
"""

        # Referral
        if treatment.requires_referral:
            referral_text = f"⚠️ **REFERRAL NEEDED**\n\nReason: {treatment.referral_reason}"
        else:
            referral_text = "✅ Can be managed at health post level"

        return diagnosis_text, treatment_text, referral_text

    except Exception as e:
        logger.error(f"Diagnosis error: {e}")
        return f"[Error: {e}]", "", ""


# ============================================================================
# STEP 4: DISPENSE - Drug Safety Check
# ============================================================================

def extract_medications_from_photo(image: Any) -> str:
    """Extract medication names from a photo."""
    if image is None:
        return ""

    try:
        hp = get_healthpost()
        medications = hp.vision.extract_medications(image)
        return "\n".join(medications)
    except Exception as e:
        logger.error(f"Medication extraction error: {e}")
        return f"[Error: {e}]"


def check_drug_interactions(
    current_meds_text: str,
    proposed_meds_text: str,
) -> Tuple[str, List[Any], List[str]]:
    """Check for drug interactions.

    Returns:
        Tuple of (formatted_output, interactions_list, dropdown_choices)
    """
    if not current_meds_text.strip() and not proposed_meds_text.strip():
        return "Enter medications to check", [], []

    try:
        hp = get_healthpost()

        # Parse medication lists
        all_meds = []

        for text in [current_meds_text, proposed_meds_text]:
            for line in text.split('\n'):
                line = line.strip().lstrip('•-* ')
                if line:
                    all_meds.append(line)

        if len(all_meds) < 2:
            return "Need at least 2 medications to check for interactions", [], []

        # Check interactions
        interactions = hp.check_drug_interactions(all_meds)

        if not interactions:
            return "✅ **No interactions found**\n\nSafe to proceed with these medications.", [], []

        # Format interactions
        output_lines = ["⚠️ **Drug Interactions Found**\n"]
        dropdown_choices = []

        for idx, interaction in enumerate(interactions):
            severity_icon = {
                "severe": "🔴 SEVERE",
                "moderate": "🟡 MODERATE",
                "mild": "🟢 MILD",
            }.get(interaction.severity, "⚪")

            output_lines.append(f"**{severity_icon}**: {interaction.drugs[0]} + {interaction.drugs[1]}")
            output_lines.append(f"   {interaction.description}")
            output_lines.append(f"   *Recommendation: {interaction.recommendation}*")
            output_lines.append("")

            # Build dropdown choice
            severity_label = interaction.severity.capitalize()
            dropdown_choices.append(f"{interaction.drugs[0]} + {interaction.drugs[1]} ({severity_label})")

        # Safety recommendation
        severe_count = sum(1 for i in interactions if i.severity == "severe")
        if severe_count > 0:
            output_lines.append("❌ **DO NOT proceed** - Severe interaction(s) detected!")
            output_lines.append("Consider alternative medications or refer to hospital.")
        else:
            output_lines.append("⚠️ **Proceed with caution** - Monitor patient closely.")

        return "\n".join(output_lines), interactions, dropdown_choices

    except Exception as e:
        logger.error(f"Interaction check error: {e}")
        return f"[Error: {e}]", [], []


def get_alternative_for_interaction(
    selected_interaction: str,
    interactions_list: List[Any],
    current_meds_text: str,
    proposed_meds_text: str,
) -> str:
    """Get an alternative medication suggestion for the selected interaction."""
    if not selected_interaction or not interactions_list:
        return ""

    try:
        hp = get_healthpost()

        # Find the selected interaction from the list
        selected_idx = None
        for idx, interaction in enumerate(interactions_list):
            severity_label = interaction.severity.capitalize()
            choice_str = f"{interaction.drugs[0]} + {interaction.drugs[1]} ({severity_label})"
            if choice_str == selected_interaction:
                selected_idx = idx
                break

        if selected_idx is None:
            return "Could not find the selected interaction"

        interaction = interactions_list[selected_idx]

        # Parse current medications
        current_meds = []
        for text in [current_meds_text]:
            for line in text.split('\n'):
                line = line.strip().lstrip('•-* ')
                if line:
                    current_meds.append(line)

        # Determine which drug to replace (prefer replacing proposed medication)
        proposed_meds = []
        for line in proposed_meds_text.split('\n'):
            line = line.strip().lstrip('•-* ')
            if line:
                proposed_meds.append(line)

        # Find which drug in the interaction is the proposed one (the one to replace)
        drug_to_replace = None
        for drug in interaction.drugs:
            drug_lower = drug.lower()
            for proposed in proposed_meds:
                if drug_lower in proposed.lower() or proposed.lower() in drug_lower:
                    drug_to_replace = drug
                    break
            if drug_to_replace:
                break

        # If no proposed med found in interaction, use the first drug in the interaction
        if not drug_to_replace:
            drug_to_replace = interaction.drugs[0]

        # Use the _suggest_alternative method from HealthPost
        # We need to provide a condition context - we'll use a generic one
        condition = "the patient's condition"

        alternative = hp._suggest_alternative(
            condition=condition,
            problematic_drug=drug_to_replace,
            interaction=interaction,
            current_meds=current_meds,
        )

        if alternative:
            return f"""💡 **Suggested Alternative**

Instead of **{drug_to_replace}**:

**{alternative}**

✓ This alternative should not interact with the other medications."""
        else:
            return f"""⚠️ **Could not suggest alternative**

Consider consulting with a healthcare provider for an appropriate alternative to **{drug_to_replace}**."""

    except Exception as e:
        logger.error(f"Alternative suggestion error: {e}")
        return f"[Error getting alternative: {e}]"


def update_interaction_ui(
    interaction_result: str,
    interactions_list: List[Any],
    dropdown_choices: List[str],
):
    """Update the interaction UI components based on results."""
    has_interactions = len(interactions_list) > 0

    return (
        gr.update(choices=dropdown_choices, value=None, visible=has_interactions),
        gr.update(visible=has_interactions),
        gr.update(value="", visible=False),
    )


# ============================================================================
# COMPLETE WORKFLOW
# ============================================================================

def run_complete_workflow(
    audio: Any,
    symptoms_text: str,
    medical_image: Any,
    image_type: str,
    patient_age: str,
    current_meds_photo: Any,
    current_meds_text: str,
) -> str:
    """Run the complete patient visit workflow."""
    try:
        hp = get_healthpost()

        # Combine symptoms from audio and text
        final_symptoms = symptoms_text.strip()
        if audio is not None:
            transcribed = hp.transcribe_symptoms(audio)
            if final_symptoms:
                final_symptoms = f"{transcribed}\n{final_symptoms}"
            else:
                final_symptoms = transcribed

        if not final_symptoms:
            return "Please provide symptoms (voice or text)"

        # Process medical images
        images = [medical_image] if medical_image is not None else []

        # Get current medications
        current_meds = []
        if current_meds_photo is not None:
            current_meds.extend(hp.extract_medications(current_meds_photo))
        if current_meds_text.strip():
            for line in current_meds_text.split('\n'):
                line = line.strip().lstrip('•-* ')
                if line:
                    current_meds.append(line)

        # Run complete workflow
        result = hp.patient_visit(
            symptoms_text=final_symptoms,
            images=images,
            existing_meds_list=current_meds,
        )

        return result.format_for_display()

    except Exception as e:
        logger.error(f"Workflow error: {e}")
        return f"[Error: {e}]"


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""

    with gr.Blocks(
        title="HealthPost - CHW Decision Support",
    ) as app:

        gr.Markdown("""
        # 🏥 HealthPost
        ### Complete Decision Support for Community Health Workers

        This tool helps you through the complete patient visit:
        **Intake → Diagnose → Prescribe → Dispense**

        ---
        """)

        with gr.Tabs():

            # ================================================================
            # TAB 1: Quick Workflow (All-in-One)
            # ================================================================
            with gr.TabItem("🚀 Quick Workflow"):
                gr.Markdown("""
                ### Complete Patient Visit
                Fill in what you have and click **Run Complete Workflow**
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 1. Patient Symptoms")
                        quick_audio = gr.Audio(
                            label="🎤 Record symptoms (optional)",
                            sources=["microphone"],
                            type="numpy",
                        )
                        quick_symptoms = gr.Textbox(
                            label="📝 Or type symptoms",
                            placeholder="Patient has fever for 3 days with headache...",
                            lines=3,
                        )
                        quick_age = gr.Textbox(
                            label="Patient age",
                            placeholder="e.g., adult, child 5 years",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### 2. Medical Image (optional)")
                        quick_image = gr.Image(
                            label="📷 Photo of skin/wound/eyes",
                            type="numpy",
                        )
                        quick_image_type = gr.Radio(
                            choices=["Skin/Rash", "Wound", "Eyes", "Other"],
                            value="Skin/Rash",
                            label="Image type",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### 3. Current Medications")
                        quick_meds_photo = gr.Image(
                            label="📷 Photo of current meds (optional)",
                            type="numpy",
                        )
                        quick_meds_text = gr.Textbox(
                            label="📝 Or list current medications",
                            placeholder="Paracetamol\nAmoxicillin",
                            lines=3,
                        )

                quick_run_btn = gr.Button(
                    "▶️ Run Complete Workflow",
                    variant="primary",
                    size="lg",
                )

                quick_output = gr.Textbox(
                    label="Complete Visit Summary",
                    lines=25,
                )

                quick_run_btn.click(
                    fn=run_complete_workflow,
                    inputs=[
                        quick_audio,
                        quick_symptoms,
                        quick_image,
                        quick_image_type,
                        quick_age,
                        quick_meds_photo,
                        quick_meds_text,
                    ],
                    outputs=quick_output,
                )

            # ================================================================
            # TAB 2: Step-by-Step Workflow
            # ================================================================
            with gr.TabItem("📋 Step-by-Step"):

                # Step 1: Intake
                with gr.Accordion("Step 1: INTAKE - Capture Symptoms", open=True):
                    with gr.Row():
                        with gr.Column():
                            intake_audio = gr.Audio(
                                label="🎤 Record patient's symptom description",
                                sources=["microphone"],
                                type="numpy",
                            )
                            transcribe_btn = gr.Button("Transcribe Audio")

                        with gr.Column():
                            intake_symptoms = gr.Textbox(
                                label="📝 Symptoms (transcribed or typed)",
                                placeholder="Patient has high fever for 3 days...",
                                lines=4,
                            )
                            intake_age = gr.Textbox(
                                label="Patient age (optional)",
                                placeholder="adult / child 5 years / infant",
                            )

                    transcribe_btn.click(
                        fn=transcribe_audio,
                        inputs=intake_audio,
                        outputs=intake_symptoms,
                    )

                # Step 2: Diagnose (Image)
                with gr.Accordion("Step 2: DIAGNOSE - Analyze Images", open=False):
                    with gr.Row():
                        with gr.Column():
                            diagnose_image = gr.Image(
                                label="📷 Upload medical image",
                                type="numpy",
                            )
                            diagnose_type = gr.Radio(
                                choices=["Skin/Rash", "Wound", "Eyes", "Other"],
                                value="Skin/Rash",
                                label="What are you photographing?",
                            )
                            analyze_btn = gr.Button("Analyze Image")

                        with gr.Column():
                            diagnose_findings = gr.Textbox(
                                label="🔍 Visual Findings",
                                lines=8,
                            )

                    analyze_btn.click(
                        fn=analyze_medical_image,
                        inputs=[diagnose_image, diagnose_type],
                        outputs=diagnose_findings,
                    )

                # Step 3: Prescribe
                with gr.Accordion("Step 3: PRESCRIBE - Get Treatment Plan", open=False):
                    with gr.Row():
                        prescribe_btn = gr.Button(
                            "Generate Diagnosis & Treatment",
                            variant="primary",
                        )

                    with gr.Row():
                        prescribe_diagnosis = gr.Markdown(label="📋 Diagnosis")
                        prescribe_treatment = gr.Markdown(label="💊 Treatment")
                        prescribe_referral = gr.Markdown(label="🏥 Referral")

                    prescribe_btn.click(
                        fn=generate_diagnosis,
                        inputs=[intake_symptoms, diagnose_findings, intake_age],
                        outputs=[prescribe_diagnosis, prescribe_treatment, prescribe_referral],
                    )

                # Step 4: Dispense
                with gr.Accordion("Step 4: DISPENSE - Safety Check", open=False):
                    gr.Markdown("Check for drug interactions before dispensing")

                    with gr.Row():
                        with gr.Column():
                            dispense_photo = gr.Image(
                                label="📷 Photo of patient's current medications",
                                type="numpy",
                            )
                            extract_meds_btn = gr.Button("Extract Medications from Photo")

                        with gr.Column():
                            dispense_current = gr.Textbox(
                                label="Current Medications",
                                placeholder="One medication per line",
                                lines=4,
                            )
                            dispense_proposed = gr.Textbox(
                                label="Proposed New Medications",
                                placeholder="One medication per line",
                                lines=4,
                            )

                    extract_meds_btn.click(
                        fn=extract_medications_from_photo,
                        inputs=dispense_photo,
                        outputs=dispense_current,
                    )

                    check_btn = gr.Button("🔍 Check Drug Interactions", variant="primary")
                    interaction_result = gr.Markdown(label="Safety Check Result")

                    # State to store interactions
                    interactions_state = gr.State([])

                    # Interaction resolution UI
                    interaction_selector = gr.Dropdown(
                        label="Select interaction to resolve",
                        choices=[],
                        visible=False,
                    )

                    get_alternative_btn = gr.Button(
                        "💡 Get Alternative Medication",
                        visible=False,
                    )

                    alternative_output = gr.Markdown(
                        label="Suggested Alternative",
                        visible=False,
                    )

                    # Wire up the check button
                    check_btn.click(
                        fn=check_drug_interactions,
                        inputs=[dispense_current, dispense_proposed],
                        outputs=[interaction_result, interactions_state, interaction_selector],
                    ).then(
                        fn=update_interaction_ui,
                        inputs=[interaction_result, interactions_state, interaction_selector],
                        outputs=[interaction_selector, get_alternative_btn, alternative_output],
                    )

                    # Wire up the get alternative button
                    get_alternative_btn.click(
                        fn=get_alternative_for_interaction,
                        inputs=[interaction_selector, interactions_state, dispense_current, dispense_proposed],
                        outputs=alternative_output,
                    ).then(
                        fn=lambda: gr.update(visible=True),
                        outputs=alternative_output,
                    )

            # ================================================================
            # TAB 3: Drug Reference
            # ================================================================
            with gr.TabItem("💊 Drug Reference"):
                gr.Markdown("""
                ### Drug Information & Interaction Checker
                Look up medications and check for interactions
                """)

                with gr.Row():
                    with gr.Column():
                        drug_search = gr.Textbox(
                            label="Search for a drug",
                            placeholder="e.g., Paracetamol, Amoxicillin",
                        )
                        search_btn = gr.Button("Search")

                    with gr.Column():
                        drug_info = gr.Markdown(label="Drug Information")

                def search_drug(query: str) -> str:
                    if not query.strip():
                        return ""
                    try:
                        hp = get_healthpost()
                        info = hp.drug_db.get_drug_info(query)
                        if not info:
                            return f"No information found for '{query}'"

                        return f"""
**{info.name}** ({info.generic_name})

**Class:** {info.drug_class}

**Common Uses:**
{chr(10).join(f'• {u}' for u in info.common_uses)}

**Contraindications:**
{chr(10).join(f'• {c}' for c in info.contraindications) if info.contraindications else '• None listed'}

**Dosages:**
{chr(10).join(f'• {k}: {v}' for k, v in info.common_doses.items())}
"""
                    except Exception as e:
                        return f"Error: {e}"

                search_btn.click(
                    fn=search_drug,
                    inputs=drug_search,
                    outputs=drug_info,
                )

                gr.Markdown("---")
                gr.Markdown("### Quick Interaction Check")

                with gr.Row():
                    interact_meds = gr.Textbox(
                        label="Enter all medications (one per line)",
                        placeholder="Paracetamol\nAmoxicillin\nMetformin",
                        lines=6,
                    )
                    interact_result = gr.Markdown(label="Interaction Results")

                # State for quick check interactions
                quick_interactions_state = gr.State([])

                # Interaction resolution UI for quick check
                quick_interaction_selector = gr.Dropdown(
                    label="Select interaction to resolve",
                    choices=[],
                    visible=False,
                )

                quick_get_alternative_btn = gr.Button(
                    "💡 Get Alternative Medication",
                    visible=False,
                )

                quick_alternative_output = gr.Markdown(
                    label="Suggested Alternative",
                    visible=False,
                )

                interact_btn = gr.Button("Check Interactions")

                # Wire up the check button
                interact_btn.click(
                    fn=lambda x: check_drug_interactions(x, ""),
                    inputs=interact_meds,
                    outputs=[interact_result, quick_interactions_state, quick_interaction_selector],
                ).then(
                    fn=update_interaction_ui,
                    inputs=[interact_result, quick_interactions_state, quick_interaction_selector],
                    outputs=[quick_interaction_selector, quick_get_alternative_btn, quick_alternative_output],
                )

                # Wire up the get alternative button
                quick_get_alternative_btn.click(
                    fn=lambda sel, interactions, meds: get_alternative_for_interaction(sel, interactions, meds, ""),
                    inputs=[quick_interaction_selector, quick_interactions_state, interact_meds],
                    outputs=quick_alternative_output,
                ).then(
                    fn=lambda: gr.update(visible=True),
                    outputs=quick_alternative_output,
                )

            # ================================================================
            # TAB 4: About
            # ================================================================
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## About HealthPost

                **HealthPost** is a complete decision support tool designed for
                Community Health Workers (CHWs) in low-resource settings.

                ### Features

                - **Voice-to-Text**: Record patient symptoms in any language
                - **Medical Image Analysis**: AI analysis of skin conditions, wounds, and eyes
                - **Diagnosis Support**: AI-assisted diagnosis with confidence scores
                - **Treatment Plans**: Evidence-based treatment recommendations
                - **Drug Safety**: Offline drug interaction checking
                - **Referral Guidance**: Know when to refer to hospital

                ### Technology

                - **MedGemma 4B**: Google's medical AI model for vision and reasoning
                - **Offline Drug Database**: WHO Essential Medicines with interaction data
                - **Edge Deployment**: Designed to run on mobile devices without internet

                ### Safety Notice

                This tool is designed to **support** clinical decision-making, not replace it.
                Always use clinical judgment and refer complex cases to higher levels of care.

                ### Credits

                Built for the MedGemma Impact Challenge 2024.

                ---

                **Version:** 0.1.0
                """)

        gr.Markdown("""
        ---
        *HealthPost - Supporting CHWs to deliver better care*
        """)

    return app


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("HealthPost - CHW Decision Support System")
    print("=" * 50)
    print("\nStarting application...")

    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for public URL
    )
