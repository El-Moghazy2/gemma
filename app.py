"""HealthPost Gradio web interface for Community Health Workers.

Supports the complete patient visit workflow:

1. **INTAKE** -- Voice/text symptom capture (MedASR).
2. **DIAGNOSE** -- Image analysis (MedGemma Vision).
3. **PRESCRIBE** -- AI-generated treatment (MedGemma Text).
4. **DISPENSE** -- Drug safety check (DDInter).
"""

import logging
import queue
import threading
from typing import Any, List, Optional, Tuple

import gradio as gr

from healthpost import Config, HealthPost

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_healthpost: Optional[HealthPost] = None


def get_healthpost() -> HealthPost:
    """Return the lazily-initialized ``HealthPost`` singleton."""
    global _healthpost
    if _healthpost is None:
        logger.info("Initializing HealthPost...")
        _healthpost = HealthPost(Config())
    return _healthpost


def _get_backend_badge(component: str) -> str:
    """Return a Markdown badge showing the active model for *component*."""
    badges = {
        "triage": "`MedGemma Text`",
        "vision": "`MedGemma Vision`",
        "voice": "`MedASR`",
    }
    return badges.get(component, "`Unknown`")


DEMO_SCENARIOS = {
    "Malaria Case": {
        "symptoms": (
            "Patient has had high fever for 3 days, shaking chills "
            "especially at night, severe headache, body aches, and "
            "sweating. Lives in malaria-endemic area. No bed net used."
        ),
        "age": "adult",
        "meds": "Metformin\nAmlodipine",
        "image_type": "Other",
    },
    "Skin Condition": {
        "symptoms": (
            "Patient presents with circular, red, raised patches on "
            "the trunk and arms. Patches have been growing for 2 weeks "
            "with itching. Central clearing visible. No fever."
        ),
        "age": "child 8 years",
        "meds": "",
        "image_type": "Skin/Rash",
    },
    "Wound + Drug Interaction": {
        "symptoms": (
            "Patient has a deep cut on the forearm from farming. Wound "
            "is 4cm long, edges are not well approximated. Some redness "
            "around edges. Patient is on warfarin for heart condition."
        ),
        "age": "adult",
        "meds": "Warfarin\nMetformin",
        "image_type": "Wound",
    },
    "Child Diarrhea": {
        "symptoms": (
            "Child has watery diarrhea for 2 days, 6-8 episodes per "
            "day. Some vomiting. Reduced appetite but still drinking. "
            "Mild fever. Eyes slightly sunken."
        ),
        "age": "child 3 years",
        "meds": "",
        "image_type": "Other",
    },
}


def load_demo_scenario(
    scenario_name: str,
) -> Tuple[str, str, str, str]:
    """Load a demo scenario into the Quick Workflow form fields."""
    if scenario_name not in DEMO_SCENARIOS:
        return "", "", "", "Skin/Rash"
    s = DEMO_SCENARIOS[scenario_name]
    return s["symptoms"], s["age"], s["meds"], s["image_type"]


def transcribe_audio(audio: Any) -> Tuple[str, str]:
    """Transcribe an audio recording of symptoms."""
    if audio is None:
        return "", ""

    try:
        hp = get_healthpost()
        text = hp.transcribe_symptoms(audio)
        source = hp.voice.source_label
        return text, f"*{source}*"
    except Exception as e:
        logger.error("Transcription error: %s", e)
        return f"[Error transcribing audio: {e}]", ""


def analyze_medical_image(image: Any, image_type: str) -> str:
    """Analyze a medical image and return formatted findings."""
    if image is None:
        return ""

    try:
        hp = get_healthpost()
        badge = _get_backend_badge("vision")

        if image_type == "Skin/Rash":
            result = hp.vision.analyze_skin_condition(image)
        elif image_type == "Wound":
            result = hp.vision.analyze_wound(image)
        else:
            findings = hp.vision.analyze_medical_image(image)
            result = {"findings": findings}

        header = f"**Powered by:** {badge}\n\n---\n\n"

        if "raw_analysis" in result:
            return header + result["raw_analysis"]
        if "findings" in result:
            return header + "\n".join(
                f"- {f}" for f in result["findings"]
            )
        return header + str(result)
    except Exception as e:
        logger.error("Image analysis error: %s", e)
        return f"[Error analyzing image: {e}]"


def generate_diagnosis(
    symptoms_text: str,
    visual_findings: str,
    patient_age: str,
):
    """Generate a diagnosis and treatment plan.

    Yields a loading indicator first, then the final result.
    """
    yield "**Analyzing...**", "", "", "*Starting...*"

    if not symptoms_text.strip():
        yield "Please provide symptoms", "", "", ""
        return

    try:
        hp = get_healthpost()
        badge = _get_backend_badge("triage")

        findings_list: List[str] = []
        if visual_findings.strip():
            for line in visual_findings.split("\n"):
                line = line.strip().lstrip("\u2022-* ")
                if line:
                    findings_list.append(line)

        diagnosis, treatment = hp.triage.diagnose_and_treat(
            symptoms=symptoms_text,
            visual_findings=findings_list,
            patient_age=patient_age if patient_age else None,
        )

        diagnosis_text = (
            f"**Powered by:** {badge}\n\n---\n\n"
            f"### {diagnosis.condition}\n\n"
            f"**Confidence:** {diagnosis.confidence:.0%}\n\n"
            f"**Supporting Evidence:**\n"
            f"{chr(10).join(f'- {e}' for e in diagnosis.supporting_evidence)}\n\n"
            f"**Differential Diagnoses:**\n"
            f"{chr(10).join(f'- {d}' for d in diagnosis.differential_diagnoses) if diagnosis.differential_diagnoses else '- None'}\n"
        )

        meds_text = "\n".join(
            f"- **{m.name}**: {m.dosage}"
            + (f" for {m.duration}" if m.duration else "")
            for m in treatment.medications
        ) if treatment.medications else "- Supportive care only"

        treatment_text = (
            f"### Medications\n{meds_text}\n\n"
            f"### Instructions\n"
            f"{chr(10).join(f'- {i}' for i in treatment.instructions)}\n\n"
            f"### Warning Signs (Return if)\n"
            f"{chr(10).join(f'- {w}' for w in treatment.warning_signs)}\n\n"
            f"**Follow-up:** {treatment.follow_up_days} days\n"
        )

        if treatment.requires_referral:
            referral_text = (
                f"### REFERRAL NEEDED\n\n"
                f"**Reason:** {treatment.referral_reason}"
            )
        else:
            referral_text = (
                "### No Referral Needed\n\n"
                "Can be managed at health post level"
            )

        trace_text = (
            "## Diagnosis Pipeline\n\n"
            f"1. Analyzed symptoms: *{symptoms_text[:80]}...*\n"
            f"2. Visual findings: {len(findings_list)} item(s)\n"
            f"3. Diagnosis: **{diagnosis.condition}** "
            f"({diagnosis.confidence:.0%})\n"
            f"4. Generated treatment plan with "
            f"{len(treatment.medications)} medication(s)\n"
            f"5. Referral: {'Yes' if treatment.requires_referral else 'No'}\n"
        )

        yield diagnosis_text, treatment_text, referral_text, trace_text
    except Exception as e:
        logger.error("Diagnosis error: %s", e)
        yield f"[Error: {e}]", "", "", ""


def extract_medications_from_photo(image: Any) -> str:
    """Extract medication names from a photo of labels."""
    if image is None:
        return ""

    try:
        hp = get_healthpost()
        medications = hp.vision.extract_medications(image)
        return "\n".join(medications)
    except Exception as e:
        logger.error("Medication extraction error: %s", e)
        return f"[Error: {e}]"


def check_drug_interactions(
    current_meds_text: str,
    proposed_meds_text: str,
) -> Tuple[str, List[Any], List[str]]:
    """Check for drug-drug interactions."""
    if not current_meds_text.strip() and not proposed_meds_text.strip():
        return "Enter medications to check", [], []

    try:
        hp = get_healthpost()

        all_meds: List[str] = []
        for text in [current_meds_text, proposed_meds_text]:
            for line in text.split("\n"):
                line = line.strip().lstrip("\u2022-* ")
                if line:
                    all_meds.append(line)

        if len(all_meds) < 2:
            return (
                "Need at least 2 medications to check for interactions",
                [], [],
            )

        interactions = hp.check_drug_interactions(all_meds)

        if not interactions:
            return (
                "### No Interactions Found\n\n"
                "Safe to proceed with these medications.",
                [], [],
            )

        output_lines = ["### Drug Interactions Found\n"]
        dropdown_choices: List[str] = []

        for interaction in interactions:
            severity_label = {
                "severe": "**SEVERE**",
                "moderate": "**MODERATE**",
                "mild": "**MILD**",
            }.get(interaction.severity, "UNKNOWN")

            output_lines.append(
                f"#### {severity_label}: "
                f"{interaction.drugs[0]} + {interaction.drugs[1]}"
            )
            output_lines.append(interaction.description)
            output_lines.append(
                f"*Recommendation: {interaction.recommendation}*\n"
            )

            sev = interaction.severity.capitalize()
            dropdown_choices.append(
                f"{interaction.drugs[0]} + "
                f"{interaction.drugs[1]} ({sev})"
            )

        severe_count = sum(
            1 for i in interactions if i.severity == "severe"
        )
        if severe_count > 0:
            output_lines.append(
                "---\n\n**DO NOT proceed** \u2014 "
                "Severe interaction(s) detected!\n"
                "Consider alternative medications or refer to hospital."
            )
        else:
            output_lines.append(
                "---\n\n**Proceed with caution** \u2014 "
                "Monitor patient closely."
            )

        return "\n".join(output_lines), interactions, dropdown_choices
    except Exception as e:
        logger.error("Interaction check error: %s", e)
        return f"[Error: {e}]", [], []


def get_alternative_for_interaction(
    selected_interaction: str,
    interactions_list: List[Any],
    current_meds_text: str,
    proposed_meds_text: str,
) -> str:
    """Suggest an alternative medication for a selected interaction."""
    if not selected_interaction or not interactions_list:
        return ""

    try:
        hp = get_healthpost()

        selected_idx = None
        for idx, interaction in enumerate(interactions_list):
            sev = interaction.severity.capitalize()
            choice = (
                f"{interaction.drugs[0]} + "
                f"{interaction.drugs[1]} ({sev})"
            )
            if choice == selected_interaction:
                selected_idx = idx
                break

        if selected_idx is None:
            return "Could not find the selected interaction"

        interaction = interactions_list[selected_idx]

        current_meds: List[str] = []
        for line in current_meds_text.split("\n"):
            line = line.strip().lstrip("\u2022-* ")
            if line:
                current_meds.append(line)

        proposed_meds: List[str] = []
        for line in proposed_meds_text.split("\n"):
            line = line.strip().lstrip("\u2022-* ")
            if line:
                proposed_meds.append(line)

        drug_to_replace = None
        for drug in interaction.drugs:
            drug_lower = drug.lower()
            for proposed in proposed_meds:
                if (
                    drug_lower in proposed.lower()
                    or proposed.lower() in drug_lower
                ):
                    drug_to_replace = drug
                    break
            if drug_to_replace:
                break

        if not drug_to_replace:
            drug_to_replace = interaction.drugs[0]

        alternative = hp._suggest_alternative(
            condition="the patient's condition",
            problematic_drug=drug_to_replace,
            interaction=interaction,
            current_meds=current_meds,
        )

        if alternative:
            return (
                f"### Suggested Alternative\n\n"
                f"Instead of **{drug_to_replace}**:\n\n"
                f"**{alternative}**\n\n"
                f"This alternative should not interact with the "
                f"other medications."
            )
        return (
            f"### Could Not Suggest Alternative\n\n"
            f"Consider consulting with a healthcare provider for an "
            f"appropriate alternative to **{drug_to_replace}**."
        )
    except Exception as e:
        logger.error("Alternative suggestion error: %s", e)
        return f"[Error getting alternative: {e}]"


def update_interaction_ui(
    interaction_result: str,
    interactions_list: List[Any],
    dropdown_choices: List[str],
):
    """Update visibility of the interaction-resolution UI components."""
    has_interactions = len(interactions_list) > 0
    return (
        gr.update(
            choices=dropdown_choices, value=None,
            visible=has_interactions,
        ),
        gr.update(visible=has_interactions),
        gr.update(value="", visible=False),
    )


def run_complete_workflow(
    audio: Any,
    symptoms_text: str,
    medical_image: Any,
    image_type: str,
    patient_age: str,
    current_meds_photo: Any,
    current_meds_text: str,
):
    """Run the complete patient visit workflow.

    Uses a background thread and queue so that pipeline progress
    streams progressively to the Gradio UI.

    Yields tuples of ``(markdown, visit_result, header_visible,
    chatbot_visible, input_row_visible)`` so the chat section
    can appear after workflow completion.
    """
    hide = gr.update(visible=False)
    yield "**Starting workflow...**", None, hide, hide, hide

    try:
        hp = get_healthpost()

        final_symptoms = symptoms_text.strip()
        if audio is not None:
            transcribed = hp.transcribe_symptoms(audio)
            if final_symptoms:
                final_symptoms = f"{transcribed}\n{final_symptoms}"
            else:
                final_symptoms = transcribed

        if not final_symptoms:
            yield "Please provide symptoms (voice or text)", None, hide, hide, hide
            return

        images = [medical_image] if medical_image is not None else []

        current_meds: List[str] = []
        if current_meds_photo is not None:
            current_meds.extend(hp.extract_medications(current_meds_photo))
        if current_meds_text.strip():
            for line in current_meds_text.split("\n"):
                line = line.strip().lstrip("\u2022-* ")
                if line:
                    current_meds.append(line)

        progress_queue: queue.Queue = queue.Queue()
        result_holder: List = [None, None]  # [result, error]

        def on_progress(step_name, detail):
            progress_queue.put(("progress", step_name, detail))

        def run_pipeline():
            try:
                result = hp.patient_visit(
                    symptoms_text=final_symptoms,
                    images=images,
                    existing_meds_list=current_meds,
                    patient_age=patient_age if patient_age else None,
                    on_progress=on_progress,
                )
                result_holder[0] = result
            except Exception as e:
                result_holder[1] = e
            finally:
                progress_queue.put(("done", None, None))

        thread = threading.Thread(target=run_pipeline)
        thread.start()

        pipeline_steps: List[str] = []

        while True:
            try:
                msg = progress_queue.get(timeout=300)
            except queue.Empty:
                break
            msg_type, key, detail = msg

            if msg_type == "done":
                break
            elif msg_type == "progress":
                pipeline_steps.append(f"**{key}**: {detail}")
                progress_md = "\n\n".join(pipeline_steps)
                yield progress_md, None, hide, hide, hide

        thread.join()

        if result_holder[1]:
            yield f"**Error:** {result_holder[1]}", None, hide, hide, hide
            return

        result = result_holder[0]
        main_output = _format_result_markdown(result, hp)
        show = gr.update(visible=True)
        yield main_output, result, show, show, show
    except Exception as e:
        logger.error("Workflow error: %s", e)
        yield f"**Error:** {e}", None, hide, hide, hide


def chat_respond(
    message: str,
    chat_history: list,
    conversation_messages: list,
    visit_result,
):
    """Handle a chat message in the post-diagnosis chat.

    Returns updated chatbot history, cleared textbox, and updated
    conversation messages state.
    """
    if not message.strip():
        return chat_history, "", conversation_messages

    if visit_result is None:
        return chat_history, "", conversation_messages

    hp = get_healthpost()

    try:
        response = hp.chat(message, conversation_messages, visit_result)
    except Exception as e:
        logger.error("Chat error: %s", e)
        response = f"Sorry, I encountered an error: {e}"

    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": response})

    if not conversation_messages:
        from healthpost.core import build_chat_system_prompt
        conversation_messages = [
            {"role": "system", "content": build_chat_system_prompt(visit_result)},
        ]
    conversation_messages.append({"role": "user", "content": message})
    conversation_messages.append({"role": "assistant", "content": response})

    return chat_history, "", conversation_messages


def _format_result_markdown(result, hp) -> str:
    """Format a ``PatientVisitResult`` as rich Markdown."""
    lines: List[str] = []

    triage_badge = _get_backend_badge("triage")
    vision_badge = _get_backend_badge("vision")
    lines.append(f"**Models:** {triage_badge} {vision_badge}\n")
    lines.append("---\n")

    lines.append(f"## Diagnosis: {result.diagnosis.condition}")
    lines.append(
        f"**Confidence:** {result.diagnosis.confidence:.0%}\n"
    )

    if result.diagnosis.supporting_evidence:
        lines.append("**Evidence:**")
        for ev in result.diagnosis.supporting_evidence[:3]:
            lines.append(f"- {ev}")
        lines.append("")

    if result.diagnosis.differential_diagnoses:
        lines.append(
            "**Consider also:** "
            + ", ".join(result.diagnosis.differential_diagnoses)
        )
        lines.append("")

    if result.diagnosis.known_symptoms:
        lines.append("**Known symptoms of this condition** *(verify with patient):*")
        for sym in result.diagnosis.known_symptoms:
            lines.append(f"- {sym}")
        lines.append("")

    lines.append("## Treatment Plan")
    if result.treatment_plan.medications:
        for med in result.treatment_plan.medications:
            dur = f" for {med.duration}" if med.duration else ""
            lines.append(f"- **{med.name}**: {med.dosage}{dur}")
    else:
        lines.append("- Supportive care")
    lines.append("")

    if result.treatment_plan.instructions:
        lines.append("**Instructions:**")
        for instr in result.treatment_plan.instructions:
            lines.append(f"- {instr}")
        lines.append("")

    if result.treatment_plan.warning_signs:
        lines.append("**Warning Signs (return if):**")
        for w in result.treatment_plan.warning_signs:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("## Drug Safety Check")
    all_meds = result.current_medications + [
        m.name for m in result.treatment_plan.medications
    ]
    if all_meds:
        lines.append(f"*Checked: {', '.join(all_meds)}*\n")

    if result.drug_interactions:
        for interaction in result.drug_interactions:
            sev = {
                "severe": "**SEVERE**",
                "moderate": "**MODERATE**",
                "mild": "MILD",
            }.get(interaction.severity, interaction.severity)
            drug_pair = " + ".join(interaction.drugs)
            desc = interaction.description[:120]
            lines.append(f"- {sev}: {drug_pair} \u2014 {desc}")
        lines.append("")

        if result.alternative_medications:
            lines.append("**Suggested Alternatives:**")
            for drug, alt in result.alternative_medications.items():
                lines.append(f"- Instead of {drug}: **{alt}**")
            lines.append("")
    else:
        lines.append("No interactions detected.\n")

    if result.is_safe_to_proceed:
        lines.append("### SAFE TO PROCEED")
    else:
        lines.append(
            "### DO NOT PROCEED \u2014 Review interactions above"
        )
    lines.append("")

    if result.needs_referral:
        lines.append(
            f"### REFERRAL NEEDED\n"
            f"**Reason:** {result.referral_reason}"
        )
    lines.append("")

    if result.treatment_plan.follow_up_days:
        lines.append(
            f"**Follow-up:** {result.treatment_plan.follow_up_days} days"
        )

    return "\n".join(lines)


def create_interface() -> gr.Blocks:
    """Build and return the Gradio ``Blocks`` application."""
    with gr.Blocks(
        title="HealthPost - CHW Decision Support",
        theme=gr.themes.Soft(),
    ) as app:

        gr.Markdown(
            "# HealthPost\n"
            "### CHW Decision Support with MedGemma\n\n"
            "AI-powered clinical workflow for Community Health Workers: "
            "**diagnosis, treatment, and drug safety** using "
            "MedGemma Vision + Text + MedASR.\n\n---"
        )

        with gr.Tabs():

            with gr.TabItem("Quick Workflow"):
                gr.Markdown(
                    "### Complete Patient Visit\n"
                    "Fill in what you have and click "
                    "**Run Complete Workflow**"
                )

                with gr.Row():
                    gr.Markdown("**Load demo scenario:**")
                with gr.Row():
                    demo_btns = {}
                    for name in DEMO_SCENARIOS:
                        demo_btns[name] = gr.Button(
                            name, size="sm", variant="secondary",
                        )

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### 1. Patient Symptoms")
                        quick_audio = gr.Audio(
                            label="Record symptoms (optional)",
                            sources=["microphone"],
                            type="numpy",
                        )
                        quick_symptoms = gr.Textbox(
                            label="Or type symptoms",
                            placeholder=(
                                "Patient has fever for 3 days "
                                "with headache..."
                            ),
                            lines=4,
                        )
                        quick_age = gr.Textbox(
                            label="Patient age",
                            placeholder="e.g., adult, child 5 years",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### 2. Medical Image (optional)")
                        quick_image = gr.Image(
                            label="Photo of skin/wound/eyes",
                            type="numpy",
                        )
                        quick_image_type = gr.Radio(
                            choices=[
                                "Skin/Rash", "Wound", "Eyes", "Other",
                            ],
                            value="Skin/Rash",
                            label="Image type",
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("#### 3. Current Medications")
                        quick_meds_photo = gr.Image(
                            label="Photo of current meds (optional)",
                            type="numpy",
                        )
                        quick_meds_text = gr.Textbox(
                            label="Or list current medications",
                            placeholder="Paracetamol\nAmoxicillin",
                            lines=3,
                        )

                quick_run_btn = gr.Button(
                    "Run Complete Workflow",
                    variant="primary",
                    size="lg",
                )

                quick_output = gr.Markdown(
                    label="Complete Visit Summary",
                )

                # --- Post-diagnosis chat ---
                visit_result_state = gr.State(None)
                chat_messages_state = gr.State([])

                chat_header = gr.Markdown(
                    "---\n### Follow-up Questions\n"
                    "Ask questions about the diagnosis, dosage, "
                    "referral criteria, etc.",
                    visible=False,
                )

                chat_chatbot = gr.Chatbot(
                    label="Ask about this diagnosis",
                    visible=False,
                    height=300,
                    type="messages",
                )
                with gr.Row(visible=False) as chat_input_row:
                    chat_textbox = gr.Textbox(
                        placeholder="e.g., What if the patient is pregnant?",
                        show_label=False,
                        scale=4,
                    )
                    chat_send_btn = gr.Button(
                        "Send", variant="primary", scale=1,
                    )

                for name, btn in demo_btns.items():
                    btn.click(
                        fn=lambda n=name: load_demo_scenario(n),
                        outputs=[
                            quick_symptoms, quick_age,
                            quick_meds_text, quick_image_type,
                        ],
                    )

                quick_run_btn.click(
                    fn=run_complete_workflow,
                    inputs=[
                        quick_audio, quick_symptoms, quick_image,
                        quick_image_type, quick_age,
                        quick_meds_photo, quick_meds_text,
                    ],
                    outputs=[
                        quick_output, visit_result_state,
                        chat_header, chat_chatbot, chat_input_row,
                    ],
                ).then(
                    fn=lambda: ([], []),
                    outputs=[chat_chatbot, chat_messages_state],
                )

                chat_inputs = [
                    chat_textbox, chat_chatbot,
                    chat_messages_state, visit_result_state,
                ]
                chat_outputs = [
                    chat_chatbot, chat_textbox, chat_messages_state,
                ]
                chat_send_btn.click(
                    fn=chat_respond,
                    inputs=chat_inputs,
                    outputs=chat_outputs,
                )
                chat_textbox.submit(
                    fn=chat_respond,
                    inputs=chat_inputs,
                    outputs=chat_outputs,
                )

            with gr.TabItem("Step-by-Step"):

                with gr.Accordion(
                    "Step 1: INTAKE - Capture Symptoms", open=True,
                ):
                    with gr.Row():
                        with gr.Column():
                            intake_audio = gr.Audio(
                                label=(
                                    "Record patient's symptom "
                                    "description"
                                ),
                                sources=["microphone"],
                                type="numpy",
                            )
                            transcribe_btn = gr.Button(
                                "Transcribe Audio",
                            )
                            transcribe_source = gr.Markdown("")

                        with gr.Column():
                            intake_symptoms = gr.Textbox(
                                label="Symptoms (transcribed or typed)",
                                placeholder=(
                                    "Patient has high fever for "
                                    "3 days..."
                                ),
                                lines=4,
                            )
                            intake_age = gr.Textbox(
                                label="Patient age (optional)",
                                placeholder=(
                                    "adult / child 5 years / infant"
                                ),
                            )

                    transcribe_btn.click(
                        fn=transcribe_audio,
                        inputs=intake_audio,
                        outputs=[intake_symptoms, transcribe_source],
                    )

                with gr.Accordion(
                    "Step 2: DIAGNOSE - Analyze Images", open=False,
                ):
                    with gr.Row():
                        with gr.Column():
                            diagnose_image = gr.Image(
                                label="Upload medical image",
                                type="numpy",
                            )
                            diagnose_type = gr.Radio(
                                choices=[
                                    "Skin/Rash", "Wound", "Eyes",
                                    "Other",
                                ],
                                value="Skin/Rash",
                                label="What are you photographing?",
                            )
                            analyze_btn = gr.Button("Analyze Image")

                        with gr.Column():
                            diagnose_findings = gr.Markdown(
                                label="Visual Findings",
                            )

                    analyze_btn.click(
                        fn=analyze_medical_image,
                        inputs=[diagnose_image, diagnose_type],
                        outputs=diagnose_findings,
                    )

                with gr.Accordion(
                    "Step 3: PRESCRIBE - Get Treatment Plan",
                    open=False,
                ):
                    prescribe_btn = gr.Button(
                        "Generate Diagnosis & Treatment",
                        variant="primary",
                    )

                    with gr.Row():
                        prescribe_diagnosis = gr.Markdown(
                            label="Diagnosis",
                        )
                        prescribe_treatment = gr.Markdown(
                            label="Treatment",
                        )
                        prescribe_referral = gr.Markdown(
                            label="Referral",
                        )

                    with gr.Accordion(
                        "Pipeline Trace", open=False,
                    ):
                        prescribe_trace = gr.Markdown(
                            label="Pipeline Trace",
                            value="*Run diagnosis to see trace*",
                            min_height=400,
                        )

                    diagnose_findings_text = gr.Textbox(visible=False)

                    prescribe_btn.click(
                        fn=generate_diagnosis,
                        inputs=[
                            intake_symptoms, diagnose_findings_text,
                            intake_age,
                        ],
                        outputs=[
                            prescribe_diagnosis, prescribe_treatment,
                            prescribe_referral, prescribe_trace,
                        ],
                    )

                with gr.Accordion(
                    "Step 4: DISPENSE - Safety Check", open=False,
                ):
                    gr.Markdown(
                        "Check for drug interactions before dispensing"
                    )

                    with gr.Row():
                        with gr.Column():
                            dispense_photo = gr.Image(
                                label=(
                                    "Photo of patient's current "
                                    "medications"
                                ),
                                type="numpy",
                            )
                            extract_meds_btn = gr.Button(
                                "Extract Medications from Photo",
                            )

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

                    check_btn = gr.Button(
                        "Check Drug Interactions", variant="primary",
                    )
                    interaction_result = gr.Markdown(
                        label="Safety Check Result",
                    )

                    interactions_state = gr.State([])

                    interaction_selector = gr.Dropdown(
                        label="Select interaction to resolve",
                        choices=[],
                        visible=False,
                    )
                    get_alternative_btn = gr.Button(
                        "Get Alternative Medication", visible=False,
                    )
                    alternative_output = gr.Markdown(
                        label="Suggested Alternative", visible=False,
                    )

                    check_btn.click(
                        fn=check_drug_interactions,
                        inputs=[dispense_current, dispense_proposed],
                        outputs=[
                            interaction_result, interactions_state,
                            interaction_selector,
                        ],
                    ).then(
                        fn=update_interaction_ui,
                        inputs=[
                            interaction_result, interactions_state,
                            interaction_selector,
                        ],
                        outputs=[
                            interaction_selector,
                            get_alternative_btn, alternative_output,
                        ],
                    )

                    get_alternative_btn.click(
                        fn=get_alternative_for_interaction,
                        inputs=[
                            interaction_selector, interactions_state,
                            dispense_current, dispense_proposed,
                        ],
                        outputs=alternative_output,
                    ).then(
                        fn=lambda: gr.update(visible=True),
                        outputs=alternative_output,
                    )

            with gr.TabItem("Drug Reference"):
                gr.Markdown(
                    "### Drug Information & Interaction Checker\n"
                    "Look up medications and check for interactions"
                )

                with gr.Row():
                    with gr.Column():
                        drug_search = gr.Textbox(
                            label="Search for a drug",
                            placeholder=(
                                "e.g., Paracetamol, Amoxicillin"
                            ),
                        )
                        search_btn = gr.Button("Search")

                    with gr.Column():
                        drug_info = gr.Markdown(
                            label="Drug Information",
                        )

                def search_drug(query: str) -> str:
                    if not query.strip():
                        return ""
                    try:
                        hp = get_healthpost()
                        info = hp.drug_db.get_drug_info(query)
                        if not info:
                            return (
                                f"No information found for '{query}'"
                            )
                        return (
                            f"### {info.name} ({info.generic_name})\n\n"
                            f"**Class:** {info.drug_class}\n\n"
                            f"**Common Uses:**\n"
                            f"{chr(10).join(f'- {u}' for u in info.common_uses)}\n\n"
                            f"**Contraindications:**\n"
                            f"{chr(10).join(f'- {c}' for c in info.contraindications) if info.contraindications else '- None listed'}\n\n"
                            f"**Dosages:**\n"
                            f"{chr(10).join(f'- {k}: {v}' for k, v in info.common_doses.items())}\n"
                        )
                    except Exception as e:
                        return f"Error: {e}"

                search_btn.click(
                    fn=search_drug,
                    inputs=drug_search,
                    outputs=drug_info,
                )

                gr.Markdown("---\n### Quick Interaction Check")

                with gr.Row():
                    interact_meds = gr.Textbox(
                        label=(
                            "Enter all medications (one per line)"
                        ),
                        placeholder=(
                            "Paracetamol\nAmoxicillin\nMetformin"
                        ),
                        lines=6,
                    )
                    interact_result = gr.Markdown(
                        label="Interaction Results",
                    )

                quick_interactions_state = gr.State([])
                quick_interaction_selector = gr.Dropdown(
                    label="Select interaction to resolve",
                    choices=[],
                    visible=False,
                )
                quick_get_alternative_btn = gr.Button(
                    "Get Alternative Medication", visible=False,
                )
                quick_alternative_output = gr.Markdown(
                    label="Suggested Alternative", visible=False,
                )

                interact_btn = gr.Button("Check Interactions")

                interact_btn.click(
                    fn=lambda x: check_drug_interactions(x, ""),
                    inputs=interact_meds,
                    outputs=[
                        interact_result, quick_interactions_state,
                        quick_interaction_selector,
                    ],
                ).then(
                    fn=update_interaction_ui,
                    inputs=[
                        interact_result, quick_interactions_state,
                        quick_interaction_selector,
                    ],
                    outputs=[
                        quick_interaction_selector,
                        quick_get_alternative_btn,
                        quick_alternative_output,
                    ],
                )

                quick_get_alternative_btn.click(
                    fn=lambda sel, interactions, meds: (
                        get_alternative_for_interaction(
                            sel, interactions, meds, "",
                        )
                    ),
                    inputs=[
                        quick_interaction_selector,
                        quick_interactions_state, interact_meds,
                    ],
                    outputs=quick_alternative_output,
                ).then(
                    fn=lambda: gr.update(visible=True),
                    outputs=quick_alternative_output,
                )

            with gr.TabItem("About"):
                gr.Markdown(
                    "## About HealthPost\n\n"
                    "**HealthPost** is a decision support tool "
                    "for Community Health Workers (CHWs) in low-resource "
                    "settings, powered by Google's MedGemma family of "
                    "medical AI models.\n\n"
                    "### Features\n\n"
                    "| Feature | Model | Description |\n"
                    "|---------|-------|-------------|\n"
                    "| Voice Intake | `MedASR` | Transcribe patient "
                    "symptoms with medical vocabulary |\n"
                    "| Image Analysis | `MedGemma Vision` | Analyze "
                    "skin conditions, wounds, eyes |\n"
                    "| Diagnosis & Treatment | `MedGemma Text` | "
                    "AI-assisted diagnosis with confidence scores |\n"
                    "| Drug Safety | `DDInter API` | Drug interaction "
                    "checking (236K+ interactions) |\n"
                    "| Referral Guidance | `Rule-based` | Know when to "
                    "refer to hospital |\n\n"
                    "### Architecture\n\n"
                    "HealthPost uses a **LangGraph pipeline** with "
                    "structured JSON output:\n\n"
                    "1. **Intake** \u2014 Capture symptoms via voice or text\n"
                    "2. **Image Analysis** \u2014 Analyze medical photos with MedGemma Vision\n"
                    "3. **Diagnosis** \u2014 JSON-structured clinical assessment via MedGemma\n"
                    "4. **Drug Safety** \u2014 Check interactions via DDInter API\n"
                    "5. **Safety Assessment** \u2014 Determine referral needs\n\n"
                    "All LLM outputs are parsed as structured JSON using "
                    "Pydantic models for reliability.\n\n"
                    "### Technical Details\n\n"
                    "- **4-bit quantization**: ~4GB VRAM, runs on "
                    "consumer GPU / Kaggle T4\n"
                    "- **DDInter API**: 236K+ drug interactions from "
                    "DrugBank, KEGG, etc.\n"
                    "- **Pydantic models**: Structured, validated data "
                    "throughout the pipeline\n"
                    "- **Gradio UI**: Mobile browser compatible\n"
                    "- **MedGemma**: Medical AI model with "
                    "multimodal support\n\n"
                    "### Safety Notice\n\n"
                    "This tool is designed to **support** clinical "
                    "decision-making, not replace it. Always use "
                    "clinical judgment and refer complex cases to "
                    "higher levels of care.\n\n"
                    "---\n\n"
                    "Built for the **MedGemma Impact Challenge 2025**."
                )

        gr.Markdown(
            "---\n*HealthPost \u2014 Supporting CHWs to deliver "
            "better care*"
        )

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("=" * 50)
    print("HealthPost - CHW Decision Support System")
    print("=" * 50)
    print("\nStarting application...")

    application = create_interface()
    application.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=args.share,
    )
