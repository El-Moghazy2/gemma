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
            f"{chr(10).join(f'- {e}' for e in diagnosis.supporting_evidence) if diagnosis.supporting_evidence else '- Based on reported symptoms'}\n\n"
            f"**Differential Diagnoses:**\n"
            f"{chr(10).join(f'- {d}' for d in diagnosis.differential_diagnoses) if diagnosis.differential_diagnoses else '- No alternative diagnoses identified'}\n\n"
            f"**Known Symptoms** *(verify with patient):*\n"
            f"{chr(10).join(f'- {s}' for s in diagnosis.known_symptoms) if diagnosis.known_symptoms else '- Verify symptoms with patient'}\n"
        )

        meds_text = "\n".join(
            f"- **{m.name}**: {m.dosage}"
            + (f" for {m.duration}" if m.duration else "")
            + (f" — {m.justification}" if m.justification else "")
            for m in treatment.medications
        ) if treatment.medications else "- Supportive care only"

        instructions_text = chr(10).join(f'- {i}' for i in treatment.instructions) if treatment.instructions else '- Follow healthcare provider guidance'
        warnings_text = chr(10).join(f'- {w}' for w in treatment.warning_signs) if treatment.warning_signs else '- Seek care if symptoms worsen or new symptoms appear'

        treatment_text = (
            f"### Medications\n{meds_text}\n\n"
            f"### Instructions\n{instructions_text}\n\n"
            f"### Warning Signs (Return if)\n{warnings_text}\n\n"
            f"**Follow-up:** {treatment.follow_up_days or 3} days\n"
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

    lines.append("**Evidence:**")
    if result.diagnosis.supporting_evidence:
        for ev in result.diagnosis.supporting_evidence[:3]:
            lines.append(f"- {ev}")
    else:
        lines.append("- Based on reported symptoms")
    lines.append("")

    if result.diagnosis.differential_diagnoses:
        lines.append(
            "**Consider also:** "
            + ", ".join(result.diagnosis.differential_diagnoses)
        )
    else:
        lines.append("**Consider also:** No alternative diagnoses identified")
    lines.append("")

    lines.append("**Known symptoms of this condition** *(verify with patient):*")
    if result.diagnosis.known_symptoms:
        for sym in result.diagnosis.known_symptoms:
            lines.append(f"- {sym}")
    else:
        lines.append("- Verify symptoms with patient")
    lines.append("")

    lines.append("## Treatment Plan")
    if result.treatment_plan.medications:
        for med in result.treatment_plan.medications:
            dur = f" for {med.duration}" if med.duration else ""
            justification = f" — {med.justification}" if med.justification else ""
            lines.append(f"- **{med.name}**: {med.dosage}{dur}{justification}")
    else:
        lines.append("- Supportive care")
    lines.append("")

    lines.append("**Instructions:**")
    if result.treatment_plan.instructions:
        for instr in result.treatment_plan.instructions:
            lines.append(f"- {instr}")
    else:
        lines.append("- Follow healthcare provider guidance")
    lines.append("")

    lines.append("**Warning Signs (return if):**")
    if result.treatment_plan.warning_signs:
        for w in result.treatment_plan.warning_signs:
            lines.append(f"- {w}")
    else:
        lines.append("- Seek care if symptoms worsen or new symptoms appear")
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

    follow_up = result.treatment_plan.follow_up_days or 3
    lines.append(f"**Follow-up:** {follow_up} days")

    return "\n".join(lines)


# ── Theme ───────────────────────────────────────────────────────────────────
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    button_primary_background_fill="linear-gradient(135deg, *primary_500, *secondary_500)",
    button_primary_text_color="white",
    block_title_text_weight="600",
    block_border_width="1px",
    block_shadow="0 1px 3px 0 rgb(0 0 0 / 0.1)",
)


def create_interface() -> gr.Blocks:
    """Build and return the Gradio ``Blocks`` application."""
    with gr.Blocks(
        title="HealthPost \u2014 AI Clinical Decision Support",
    ) as app:

        # Header
        gr.Markdown(
            """
            <div style="text-align:center; padding: 1.5rem 0;">
                <h1 style="margin:0;">HealthPost</h1>
                <p style="opacity:0.7; margin:0.25rem 0 0.5rem;">
                    AI-powered clinical decision support &mdash; diagnosis, treatment &amp; drug safety
                </p>
                <span style="background:rgba(0,128,128,0.1); padding:4px 12px; border-radius:20px; font-size:0.85rem;">
                    Powered by MedGemma
                </span>
                &nbsp;
                <span style="background:rgba(0,128,128,0.1); padding:4px 12px; border-radius:20px; font-size:0.85rem;">
                    MedGemma Impact Challenge 2025
                </span>
            </div>
            """
        )

        with gr.Tabs():

            with gr.Tab("Clinical Workspace"):

                # Demo loader
                with gr.Group():
                    demo_dropdown = gr.Dropdown(
                        choices=list(DEMO_SCENARIOS.keys()),
                        label="Load a demo scenario (optional)",
                        value=None,
                        interactive=True,
                    )

                # Patient intake
                with gr.Group():
                    gr.Markdown("### Patient Information")
                    quick_symptoms = gr.Textbox(
                        label="Symptoms",
                        placeholder=(
                            "Describe the patient's symptoms in detail..."
                        ),
                        lines=4,
                    )
                    with gr.Row():
                        quick_age = gr.Textbox(
                            label="Patient Age",
                            placeholder="e.g., adult, child 5 years",
                            scale=1,
                        )
                        quick_audio = gr.Audio(
                            label="Or record symptoms (MedASR)",
                            sources=["microphone"],
                            type="numpy",
                            scale=1,
                        )

                # Optional attachments
                with gr.Row():
                    with gr.Accordion(
                        "Add Medical Image (optional)", open=False,
                    ):
                        quick_image = gr.Image(
                            label="Upload a medical photo",
                            type="numpy",
                        )
                        quick_image_type = gr.Radio(
                            choices=[
                                "Skin/Rash", "Wound", "Eyes", "Other",
                            ],
                            value="Skin/Rash",
                            label="Image Type",
                        )
                    with gr.Accordion(
                        "Current Medications (optional)", open=False,
                    ):
                        quick_meds_photo = gr.Image(
                            label="Upload photo of medications",
                            type="numpy",
                        )
                        quick_meds_text = gr.Textbox(
                            label="Or type medication names",
                            placeholder="One per line",
                            lines=3,
                        )

                # Run button
                quick_run_btn = gr.Button(
                    "Run Complete Workflow",
                    variant="primary",
                    size="lg",
                )

                # Diagnostic output
                quick_output = gr.Markdown(
                    label="Diagnostic Report",
                )

                # --- Post-diagnosis chat ---
                visit_result_state = gr.State(None)
                chat_messages_state = gr.State([])

                chat_header = gr.Markdown(
                    "### Follow-up Questions\n"
                    "_Ask questions about the diagnosis, dosage, "
                    "referral criteria, etc._",
                    visible=False,
                )

                chat_chatbot = gr.Chatbot(
                    label="Ask about this diagnosis",
                    visible=False,
                    height=300,
                    layout="bubble",
                )
                with gr.Row(visible=False) as chat_input_row:
                    chat_textbox = gr.Textbox(
                        placeholder="e.g., What if the patient is pregnant?",
                        show_label=False,
                        scale=9,
                    )
                    chat_send_btn = gr.Button(
                        "Send", variant="primary", scale=1,
                    )

                demo_dropdown.change(
                    fn=load_demo_scenario,
                    inputs=demo_dropdown,
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

            with gr.Tab("Architecture & About"):
                gr.Markdown(
                    """
## System Architecture

HealthPost orchestrates **five specialised clinical AI modules** into one seamless workflow:

| Step | Module | Purpose |
|------|--------|---------|
| 1 | **MedASR** | Voice \u2192 structured symptom text |
| 2 | **MedGemma Vision** | Analyse medical images (skin, wounds, eyes) |
| 3 | **MedGemma Text** | Differential diagnosis + treatment plan |
| 4 | **DDInter API** | Drug-drug interaction safety check |
| 5 | **Referral Engine** | Rule-based escalation guidance |

---

### Why HealthPost?

- **One-click workflow** \u2014 all modules run automatically in sequence
- **Offline-ready** \u2014 designed for low-connectivity health posts
- **CHW-friendly** \u2014 plain-language output, no medical jargon
- **Safety-first** \u2014 drug interactions checked before any prescription

---

### Important Disclaimer

> This tool is a **decision-support system** for Community Health Workers.
> It does **not** replace professional medical judgment.
> All AI-generated recommendations should be reviewed by a qualified clinician.

---

*Built for the MedGemma Impact Challenge 2025*
                    """
                )

        # Footer
        gr.Markdown(
            """
            <div style="text-align:center; padding:1rem 0; opacity:0.6; font-size:0.85rem;">
                <strong>HealthPost</strong> \u2014 Supporting CHWs to deliver better care &nbsp;|&nbsp;
                Built for the MedGemma Impact Challenge 2025
            </div>
            """
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
        theme=theme,
    )
