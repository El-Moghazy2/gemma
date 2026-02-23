"""HealthPost Gradio web interface for Community Health Workers.

Supports the complete patient visit workflow:

1. **INTAKE** -- Voice/text symptom capture (MedASR / Whisper fallback).
2. **DIAGNOSE** -- Image analysis (MedGemma Vision).
3. **PRESCRIBE** -- AI-generated treatment (MedGemma Text).
4. **DISPENSE** -- Drug safety check (DDInter).
"""

import os

# Use persistent storage for model cache (survives Space restarts on T4)
os.environ["HF_HOME"] = "/data/.huggingface"

import logging
from typing import Any, List, Optional, Tuple

import gradio as gr
from healthpost import Config, HealthPost

try:
    import spaces
except ImportError:
    class spaces:  # type: ignore[no-redef]
        """No-op fallback so the app runs without HF Spaces infrastructure."""
        @staticmethod
        def GPU(duration: int = 60):
            return lambda fn: fn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_healthpost: Optional[HealthPost] = None


def get_healthpost() -> HealthPost:
    """Return the lazily-initialized ``HealthPost`` singleton.

    Returns:
        The shared ``HealthPost`` instance, created on first call.
    """
    global _healthpost
    if _healthpost is None:
        logger.info("Initializing HealthPost...")
        _healthpost = HealthPost(Config())
    return _healthpost


def _get_backend_badge(component: str) -> str:
    """Return a Markdown badge showing the active model for *component*.

    Args:
        component: Subsystem key (``"triage"``, ``"vision"``, or
            ``"voice"``).

    Returns:
        Inline Markdown code span identifying the model.
    """
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
    },
    "Skin Condition": {
        "symptoms": (
            "Patient presents with circular, red, raised patches on "
            "the trunk and arms. Patches have been growing for 2 weeks "
            "with itching. Central clearing visible. No fever."
        ),
        "age": "child 8 years",
        "meds": "",
    },
    "Wound + Drug Interaction": {
        "symptoms": (
            "Patient has a deep cut on the forearm from farming. Wound "
            "is 4cm long, edges are not well approximated. Some redness "
            "around edges. Patient is on warfarin for heart condition."
        ),
        "age": "adult",
        "meds": "Warfarin\nMetformin",
    },
    "Child Diarrhea": {
        "symptoms": (
            "Child has watery diarrhea for 2 days, 6-8 episodes per "
            "day. Some vomiting. Reduced appetite but still drinking. "
            "Mild fever. Eyes slightly sunken."
        ),
        "age": "child 3 years",
        "meds": "",
    },
}


def load_demo_scenario(
    scenario_name: str,
) -> Tuple[str, str, str]:
    """Load a demo scenario into the Quick Workflow form fields.

    Args:
        scenario_name: Key in ``DEMO_SCENARIOS``.

    Returns:
        Tuple of ``(symptoms, age, medications)`` strings.
    """
    if scenario_name not in DEMO_SCENARIOS:
        return "", "", ""
    s = DEMO_SCENARIOS[scenario_name]
    return s["symptoms"], s["age"], s["meds"]


@spaces.GPU(duration=300)
def _transcribe_audio_gpu(audio: Any) -> Tuple[str, str]:
    """GPU-accelerated transcription helper.

    Args:
        audio: NumPy audio array from Gradio.

    Returns:
        Tuple of ``(transcribed_text, source_label_markdown)``.
    """
    hp = get_healthpost()
    text = hp.transcribe_symptoms(audio)
    source = hp.voice.source_label
    return text, f"*{source}*"


def transcribe_audio(audio: Any) -> Tuple[str, str]:
    """Transcribe an audio recording of symptoms.

    Args:
        audio: NumPy audio array from Gradio, or ``None``.

    Returns:
        Tuple of ``(transcribed_text, source_label_markdown)``.
    """
    if audio is None:
        return "", ""

    try:
        return _transcribe_audio_gpu(audio)
    except Exception as e:
        logger.error("Transcription error: %s", e)
        return f"[Error transcribing audio: {e}]", ""


@spaces.GPU(duration=300)
def _analyze_image_gpu(image: Any, image_type: str) -> dict:
    """GPU-accelerated image analysis helper.

    Args:
        image: NumPy image array from Gradio.
        image_type: One of ``"Skin/Rash"``, ``"Wound"``, or
            ``"General"``.

    Returns:
        Dict with analysis results keyed by finding type.
    """
    hp = get_healthpost()
    if image_type == "Skin/Rash":
        return hp.vision.analyze_skin_condition(image)
    elif image_type == "Wound":
        return hp.vision.analyze_wound(image)
    else:
        findings = hp.vision.analyze_medical_image(image)
        return {"findings": findings}


def analyze_medical_image(image: Any, image_type: str) -> str:
    """Analyze a medical image and return formatted findings.

    Args:
        image: NumPy image array from Gradio, or ``None``.
        image_type: Category label for the image.

    Returns:
        Markdown-formatted analysis string.
    """
    if image is None:
        return ""

    try:
        badge = _get_backend_badge("vision")

        result = _analyze_image_gpu(image, image_type)

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


@spaces.GPU(duration=300)
def _diagnose_gpu(symptoms_text: str, findings_list: List[str], patient_age: Optional[str]):
    """GPU-accelerated diagnosis helper.

    Args:
        symptoms_text: Patient symptom description.
        findings_list: Visual findings from image analysis.
        patient_age: Age description, or ``None``.

    Returns:
        Tuple of ``(Diagnosis, TreatmentPlan)``.
    """
    hp = get_healthpost()
    return hp.triage.diagnose_and_treat(
        symptoms=symptoms_text,
        visual_findings=findings_list,
        patient_age=patient_age,
    )


def generate_diagnosis(
    symptoms_text: str,
    visual_findings: str,
    patient_age: str,
):
    """Generate a diagnosis and treatment plan.

    Yields a loading indicator first, then the final result.

    Args:
        symptoms_text: Patient symptom description.
        visual_findings: Newline-separated visual findings.
        patient_age: Age description string.

    Yields:
        Tuples of ``(diagnosis_md, treatment_md, referral_md,
        trace_md)``.
    """
    yield "**Analyzing...**", "", "", "*Starting...*"

    if not symptoms_text.strip():
        yield "Please provide symptoms", "", "", ""
        return

    try:
        badge = _get_backend_badge("triage")

        findings_list: List[str] = []
        if visual_findings.strip():
            for line in visual_findings.split("\n"):
                line = line.strip().lstrip("\u2022-* ")
                if line:
                    findings_list.append(line)

        diagnosis, treatment = _diagnose_gpu(
            symptoms_text, findings_list,
            patient_age if patient_age else None,
        )

        evidence_lines = (
            "\n".join(f"- {e}" for e in diagnosis.supporting_evidence)
            if diagnosis.supporting_evidence
            else "- Based on reported symptoms"
        )
        differential_lines = (
            "\n".join(f"- {d}" for d in diagnosis.differential_diagnoses)
            if diagnosis.differential_diagnoses
            else "- No alternative diagnoses identified"
        )
        known_symptoms_lines = (
            "\n".join(f"- {s}" for s in diagnosis.known_symptoms)
            if diagnosis.known_symptoms
            else "- Verify symptoms with patient"
        )

        diagnosis_text = (
            f"**Powered by:** {badge}\n\n---\n\n"
            f"### {diagnosis.condition}\n\n"
            f"**Confidence:** {diagnosis.confidence:.0%}\n\n"
            f"**Supporting Evidence:**\n"
            f"{evidence_lines}\n\n"
            f"**Differential Diagnoses:**\n"
            f"{differential_lines}\n\n"
            f"**Known Symptoms** *(verify with patient):*\n"
            f"{known_symptoms_lines}\n"
        )

        meds_text = "\n".join(
            f"- **{m.name}**: {m.dosage}"
            + (f" for {m.duration}" if m.duration else "")
            + (f" — {m.justification}" if m.justification else "")
            for m in treatment.medications
        ) if treatment.medications else "- Supportive care only"

        instructions_text = (
            "\n".join(f"- {i}" for i in treatment.instructions)
            if treatment.instructions
            else "- Follow healthcare provider guidance"
        )
        warnings_text = (
            "\n".join(f"- {w}" for w in treatment.warning_signs)
            if treatment.warning_signs
            else "- Seek care if symptoms worsen or new symptoms appear"
        )

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


@spaces.GPU(duration=300)
def _extract_medications_gpu(image: Any) -> List[str]:
    """GPU-accelerated medication extraction helper.

    Args:
        image: NumPy image array of medication labels.

    Returns:
        List of extracted medication name strings.
    """
    hp = get_healthpost()
    return hp.vision.extract_medications(image)


def extract_medications_from_photo(image: Any) -> str:
    """Extract medication names from a photo of labels.

    Args:
        image: NumPy image array, or ``None``.

    Returns:
        Newline-separated medication names, or an error string.
    """
    if image is None:
        return ""

    try:
        medications = _extract_medications_gpu(image)
        return "\n".join(medications)
    except Exception as e:
        logger.error("Medication extraction error: %s", e)
        return f"[Error: {e}]"


def check_drug_interactions(
    current_meds_text: str,
    proposed_meds_text: str,
) -> Tuple[str, List[Any], List[str]]:
    """Check for drug-drug interactions.

    Args:
        current_meds_text: Newline-separated current medication names.
        proposed_meds_text: Newline-separated proposed medication names.

    Returns:
        Tuple of ``(markdown_report, interactions_list,
        dropdown_choices)``.
    """
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
    """Suggest an alternative medication for a selected interaction.

    Args:
        selected_interaction: Dropdown label of the chosen interaction.
        interactions_list: Full list of interaction objects.
        current_meds_text: Newline-separated current medication names.
        proposed_meds_text: Newline-separated proposed medication names.

    Returns:
        Markdown-formatted alternative suggestion.
    """
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
    """Update visibility of the interaction-resolution UI components.

    Args:
        interaction_result: Markdown report from the interaction check.
        interactions_list: List of interaction objects.
        dropdown_choices: Labels for the interaction dropdown.

    Returns:
        Tuple of Gradio updates for ``(dropdown, button, alt_output)``.
    """
    has_interactions = len(interactions_list) > 0
    return (
        gr.update(
            choices=dropdown_choices, value=None,
            visible=has_interactions,
        ),
        gr.update(visible=has_interactions),
        gr.update(value="", visible=False),
    )


@spaces.GPU(duration=300)
def _run_pipeline_gpu(
    symptoms: str,
    images: List[Any],
    meds: List[str],
    age: Optional[str],
):
    """GPU-accelerated patient visit pipeline.

    Args:
        symptoms: Patient symptom description.
        images: List of medical image arrays.
        meds: Current medication names.
        age: Patient age description, or ``None``.

    Returns:
        ``PatientVisitResult`` from the complete workflow.
    """
    hp = get_healthpost()
    return hp.patient_visit(
        symptoms_text=symptoms,
        images=images,
        existing_meds_list=meds,
        patient_age=age,
    )


def run_complete_workflow(
    audio: Any,
    symptoms_text: str,
    medical_image: Any,
    patient_age: str,
    current_meds_photo: Any,
    current_meds_text: str,
):
    """Run the complete patient visit workflow.

    Args:
        audio: NumPy audio array, or ``None``.
        symptoms_text: Text description of symptoms.
        medical_image: NumPy image array, or ``None``.
        patient_age: Age description string.
        current_meds_photo: NumPy image of medication labels, or
            ``None``.
        current_meds_text: Newline-separated medication names.

    Yields:
        Tuples of ``(markdown, visit_result, header_visible,
        chatbot_visible, input_row_visible)``.
    """
    hide = gr.update(visible=False)
    yield "**Starting workflow...**", None, hide, hide, hide

    try:
        final_symptoms = symptoms_text.strip()
        if audio is not None:
            transcribed, _ = _transcribe_audio_gpu(audio)
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
            current_meds.extend(_extract_medications_gpu(current_meds_photo))
        if current_meds_text.strip():
            for line in current_meds_text.split("\n"):
                line = line.strip().lstrip("\u2022-* ")
                if line:
                    current_meds.append(line)

        yield "**Running AI analysis...**", None, hide, hide, hide

        result = _run_pipeline_gpu(
            final_symptoms, images, current_meds,
            patient_age if patient_age else None,
        )

        hp = get_healthpost()
        main_output = _format_result_markdown(result, hp)
        show = gr.update(visible=True)
        yield main_output, result, show, show, show
    except Exception as e:
        logger.error("Workflow error: %s", e)
        yield f"**Error:** {e}", None, hide, hide, hide


@spaces.GPU(duration=300)
def _chat_respond_gpu(message: str, conversation_messages: list, visit_result: Any) -> str:
    """GPU-accelerated chat response helper.

    Args:
        message: The health worker's new question.
        conversation_messages: Previous conversation role/content dicts.
        visit_result: Completed ``PatientVisitResult``.

    Returns:
        The assistant's response text.
    """
    hp = get_healthpost()
    return hp.chat(message, conversation_messages, visit_result)


def chat_respond(
    message: str,
    chat_history: list,
    conversation_messages: list,
    visit_result: Any,
):
    """Handle a chat message in the post-diagnosis chat.

    Args:
        message: The health worker's new question.
        chat_history: Gradio chatbot message history.
        conversation_messages: Full conversation role/content dicts.
        visit_result: Completed ``PatientVisitResult``, or ``None``.

    Returns:
        Tuple of ``(updated_history, cleared_textbox,
        updated_messages)``.
    """
    if not message.strip():
        return chat_history, "", conversation_messages

    if visit_result is None:
        return chat_history, "", conversation_messages

    try:
        response = _chat_respond_gpu(message, conversation_messages, visit_result)
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


def _format_result_markdown(result: Any, hp: Any) -> str:
    """Format a ``PatientVisitResult`` as rich Markdown.

    Args:
        result: Completed ``PatientVisitResult``.
        hp: ``HealthPost`` instance (used for badge labels).

    Returns:
        Multi-section Markdown string for the diagnostic report.
    """
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
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=gr.themes.GoogleFont("Inter"),
    font_mono=gr.themes.GoogleFont("JetBrains Mono"),
).set(
    body_background_fill="#F5F7FA",
    block_background_fill="#FFFFFF",
    panel_background_fill="#FFFFFF",
    block_radius="16px",
    block_shadow="0 1px 3px 0 rgba(0,102,255,0.06), 0 1px 2px -1px rgba(0,102,255,0.04)",
    button_primary_background_fill="linear-gradient(135deg, #0066FF, #5C6CEB)",
    button_primary_background_fill_hover="linear-gradient(135deg, #0055DD, #4C5CDB)",
    button_primary_text_color="#FFFFFF",
    block_title_text_weight="600",
    block_title_background_fill="transparent",
    block_title_background_fill_dark="transparent",
    block_label_background_fill="transparent",
    block_label_background_fill_dark="transparent",
    block_border_width="1px",
    input_border_color_focus="#0066FF",
    input_background_fill="#FAFBFC",
    input_shadow_focus="0 0 0 3px rgba(0,102,255,0.12)",
    chatbot_text_size="14px",
)

# ── Font loading ────────────────────────────────────────────────────────────
CUSTOM_HEAD = '<meta name="viewport" content="width=device-width, initial-scale=1.0">'

# ── Gradient header HTML ────────────────────────────────────────────────────
HEADER_HTML = """
<div id="hp-header" style="
    background: linear-gradient(135deg, #0066FF, #5C6CEB);
    padding: 3.5rem 2rem;
    margin-bottom: 1rem;
">
    <!-- Decorative blur circles -->
    <div style="
        position: absolute; top: -40px; right: -40px;
        width: 200px; height: 200px;
        background: rgba(255,255,255,0.2);
        border-radius: 50%;
        filter: blur(48px);
    "></div>
    <div style="
        position: absolute; bottom: -30px; left: -30px;
        width: 150px; height: 150px;
        background: rgba(255,255,255,0.15);
        border-radius: 50%;
        filter: blur(48px);
    "></div>
    <div style="text-align: center; position: relative; z-index: 1;">
        <!-- Icon -->
        <div style="
            display: inline-flex; align-items: center; justify-content: center;
            width: 64px; height: 64px;
            background: rgba(255,255,255,0.18);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border-radius: 16px;
            margin-bottom: 1rem;
        ">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
        </div>
        <h1 style="
            margin: 0; color: #FFFFFF;
            font-family: 'Inter', sans-serif;
            font-weight: 800; font-size: 2.5rem;
        ">HealthPost</h1>
        <p style="
            color: rgba(255,255,255,0.85);
            margin: 0.75rem 0 1.5rem;
            font-size: 1rem;
        ">AI-powered clinical decision support &mdash; diagnosis, treatment &amp; drug safety</p>
        <div style="display: flex; gap: 0.75rem; justify-content: center; flex-wrap: wrap;">
            <span style="
                background: rgba(255,255,255,0.18);
                backdrop-filter: blur(8px);
                -webkit-backdrop-filter: blur(8px);
                padding: 6px 16px;
                border-radius: 20px;
                font-size: 0.85rem;
                color: #FFFFFF;
                font-weight: 500;
            ">Powered by MedGemma</span>
            <span style="
                background: rgba(255,255,255,0.18);
                backdrop-filter: blur(8px);
                -webkit-backdrop-filter: blur(8px);
                padding: 6px 16px;
                border-radius: 20px;
                font-size: 0.85rem;
                color: #FFFFFF;
                font-weight: 500;
            ">MedGemma Impact Challenge 2025</span>
        </div>
    </div>
</div>
"""

# ── Styled footer HTML ─────────────────────────────────────────────────────
FOOTER_HTML = """
<div style="
    text-align: center;
    padding: 1.25rem 0 0.75rem;
    margin-top: 1.5rem;
    border-top: 1px solid #E8E9F3;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: #6B7280;
">
    <strong style="color: #1A1F2E;">HealthPost</strong> &mdash;
    Supporting CHWs to deliver better care &nbsp;|&nbsp;
    Built for the MedGemma Impact Challenge 2025
</div>
"""

# ── Custom CSS ──────────────────────────────────────────────────────────────
CUSTOM_CSS = """
/* ── Global ─────────────────────────────────────────────────────────────── */
html, body {
    overflow-x: hidden !important;
}
.gradio-container {
    max-width: 100% !important;
    margin: 0 auto !important;
    background: #F5F7FA !important;
    font-family: 'Inter', sans-serif !important;
    overflow-x: hidden !important;
}
/* Force ALL inner Gradio wrappers to full width */
.gradio-container > div,
.gradio-container > div > div,
.gradio-container > div > div > div {
    width: 100% !important;
    max-width: 100% !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}
/* ── Full-width header (break out of parent padding) ──────────────────── */
#hp-header-wrap {
    width: 100vw !important;
    margin-left: calc(-50vw + 50%) !important;
    margin-right: calc(-50vw + 50%) !important;
    max-width: none !important;
    padding: 0 !important;
    overflow: visible !important;
}
#hp-header {
    position: relative !important;
    border-radius: 0 !important;
    margin: 0 !important;
    width: 100% !important;
    box-sizing: border-box !important;
}

/* ── Content area: 2/3 width, centered ─────────────────────────────────── */
#main-tabs, #hp-footer {
    width: 80% !important;
    max-width: 80% !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
h1, h2, h3, h4, h5 {
    font-family: 'Inter', sans-serif !important;
}

/* ── Remove ALL label/header highlight backgrounds ─────────────────────── */
label, label span,
.gr-group label, .gr-panel label,
.gr-block label, .gr-block label span,
.block-label,
span.svelte-1gfkn6j,
.gr-input-label, .gr-box > label,
[data-testid] label, [data-testid] label span,
.gradio-container label, .gradio-container label span {
    background: none !important;
    background-color: transparent !important;
    background-image: none !important;
    color: #1A1F2E !important;
}
/* Remove highlighted bar behind section HTML inside groups */
.gr-group > .gr-html:first-child,
.gr-group > div > .gr-html,
.gr-group .prose,
.gr-group .prose p,
.gr-group > div,
.gr-group > div > div,
.gr-group > div > div > div,
.gr-group > div:first-child > div {
    background: none !important;
    background-color: transparent !important;
    background-image: none !important;
}
/* Strip colored backgrounds from labels/headers inside groups */
.gr-group > div > div,
.gr-group .prose,
.gr-group .prose p,
.gr-group label,
.gr-group label span {
    background-color: transparent !important;
}
.gr-group {
    background-color: #FFFFFF !important;
}
.gr-group textarea,
.gr-group input[type="text"],
.gr-group input[type="number"] {
    background-color: #FAFBFC !important;
}
.gr-group button {
    background-color: revert !important;
}
/* Dropdown menu needs a solid background */
.gr-group ul[role="listbox"],
.gr-group [data-testid="dropdown"] ul,
.gr-group .dropdown-menu,
.gr-group ul.options,
.gr-group div[role="listbox"],
.gr-group .secondary-wrap ul,
.gr-group .wrap ul {
    background-color: #FFFFFF !important;
}
.gr-group ul[role="listbox"] li,
.gr-group [data-testid="dropdown"] ul li,
.gr-group .dropdown-menu li,
.gr-group ul.options li,
.gr-group div[role="listbox"] > div,
.gr-group .secondary-wrap ul li,
.gr-group .wrap ul li {
    background-color: #FFFFFF !important;
    color: #1A1F2E !important;
}
.gr-group ul[role="listbox"] li:hover,
.gr-group [data-testid="dropdown"] ul li:hover,
.gr-group ul.options li:hover,
.gr-group div[role="listbox"] > div:hover,
.gr-group .secondary-wrap ul li:hover,
.gr-group .wrap ul li:hover {
    background-color: #E6F2FF !important;
}

/* ── Tabs (frosted glass) ───────────────────────────────────────────────── */
.tabs > .tab-nav {
    background: rgba(241,243,248,0.6) !important;
    backdrop-filter: blur(8px) !important;
    -webkit-backdrop-filter: blur(8px) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid #E8E9F3 !important;
}
.tabs > .tab-nav > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease !important;
    border: none !important;
    background: transparent !important;
    color: #6B7280 !important;
}
.tabs > .tab-nav > button.selected {
    background: #FFFFFF !important;
    color: #1A1F2E !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
    font-weight: 600 !important;
}

/* ── Panels / Groups ────────────────────────────────────────────────────── */
.gr-group, .gr-panel, .gr-block {
    background: #FFFFFF !important;
    border-radius: 16px !important;
    box-shadow: 0 1px 3px 0 rgba(0,102,255,0.06), 0 1px 2px -1px rgba(0,102,255,0.04) !important;
    transition: box-shadow 0.25s ease !important;
    border-color: #E8E9F3 !important;
}
.gr-group:hover, .gr-panel:hover {
    box-shadow: 0 4px 24px -4px rgba(0,102,255,0.08), 0 1px 4px rgba(26,31,46,0.04) !important;
}

/* ── Run Workflow button ────────────────────────────────────────────────── */
#run-workflow-btn {
    height: 56px !important;
    background: linear-gradient(135deg, #0066FF, #5C6CEB) !important;
    color: #FFFFFF !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border-radius: 14px !important;
    border: none !important;
    box-shadow: 0 4px 14px -2px rgba(0,102,255,0.35) !important;
    transition: all 0.2s ease !important;
}
#run-workflow-btn:hover {
    box-shadow: 0 6px 20px -2px rgba(0,102,255,0.45) !important;
    transform: scale(1.01);
}

/* ── Chat send button ───────────────────────────────────────────────────── */
#chat-send-btn {
    background: linear-gradient(135deg, #0066FF, #5C6CEB) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
}

/* ── Upload areas ───────────────────────────────────────────────────────── */
.upload-container, [data-testid="image"] .image-container {
    border: 2px dashed #E8E9F3 !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}
.upload-container:hover, [data-testid="image"] .image-container:hover {
    border-color: rgba(0,102,255,0.3) !important;
    background: rgba(230,242,255,0.3) !important;
}

/* ── Chatbot bubbles ────────────────────────────────────────────────────── */
#follow-up-chatbot .message.user .message-content {
    background: linear-gradient(135deg, #0066FF, #5C6CEB) !important;
    color: #FFFFFF !important;
    border-radius: 16px 16px 4px 16px !important;
}
#follow-up-chatbot .message.bot .message-content {
    background: #E6F2FF !important;
    color: #1A1F2E !important;
    border-radius: 16px 16px 16px 4px !important;
}

/* ── Diagnostic output (elevated shadow) ────────────────────────────────── */
#diagnostic-output {
    box-shadow: 0 8px 32px -8px rgba(0,102,255,0.12), 0 2px 8px rgba(26,31,46,0.04) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
}
#diagnostic-output h2 {
    font-family: 'Inter', sans-serif !important;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #E6F2FF;
    margin-top: 1.5rem;
}

/* ── About tab ──────────────────────────────────────────────────────────── */
#about-content {
    background: #FFFFFF !important;
    border-radius: 16px !important;
    padding: 2rem !important;
}
#about-content table {
    width: 100%;
    border-collapse: collapse;
}
#about-content table th {
    background: #E6F2FF !important;
    padding: 10px 14px;
    text-align: left;
    font-weight: 600;
    color: #1A1F2E;
}
#about-content table td {
    padding: 10px 14px;
    border-bottom: 1px solid #E8E9F3;
}
#about-content code {
    background: rgba(0,102,255,0.08);
    color: #0066FF;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.85em;
}

/* ── Inputs ─────────────────────────────────────────────────────────────── */
textarea, input[type="text"], input[type="number"], .gr-input {
    border-radius: 12px !important;
    border: 1px solid #E0E3EA !important;
    padding: 10px 14px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    background: #FAFBFC !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
textarea:focus, input:focus, .gr-input:focus {
    border-color: #0066FF !important;
    box-shadow: 0 0 0 3px rgba(0,102,255,0.12) !important;
    background: #FFFFFF !important;
}

/* ── Audio component ───────────────────────────────────────────────────── */
.audio-container, [data-testid="audio"] {
    border-radius: 12px !important;
}
[data-testid="audio"] .wrap {
    border: 2px dashed #E0E3EA !important;
    border-radius: 12px !important;
    background: #FAFBFC !important;
}
[data-testid="audio"] .tab-nav button,
[data-testid="audio"] .tab-nav button.selected {
    border: none !important;
    border-bottom: none !important;
    box-shadow: none !important;
}

/* ── Accordion ──────────────────────────────────────────────────────────── */
.gr-accordion {
    border-radius: 16px !important;
    border-color: #E8E9F3 !important;
}
.gr-accordion > .label-wrap {
    background-color: #1A1F2E !important;
    color: #FFFFFF !important;
    font-size: 1.05rem !important;
    padding: 12px 16px !important;
    border-radius: 16px !important;
}
.gr-accordion > .label-wrap,
.gr-accordion > .label-wrap *,
.gr-accordion > .label-wrap span,
.gr-accordion > .label-wrap .icon {
    color: #FFFFFF !important;
    background-color: #1A1F2E !important;
}

/* ── Attachments row: equal height when open, natural when closed ──────── */
#attachments-row {
    align-items: flex-start !important;
}
#attachments-row > div {
    display: flex !important;
    flex-direction: column !important;
}
#attachments-row .gr-accordion.open {
    flex: 1 !important;
}
/* When accordion is open, stretch both columns equally */
#attachments-row:has(.open) {
    align-items: stretch !important;
}
#attachments-row:has(.open) .gr-accordion {
    height: 100% !important;
}

/* ── Section header spacing ────────────────────────────────────────────── */
.gr-group h3, .gr-panel h3 {
    margin-top: 0.5rem !important;
}

/* ── Group inner padding ───────────────────────────────────────────────── */
.gr-group {
    padding: 1.25rem !important;
}

/* ── Tab content breathing room ────────────────────────────────────────── */
.tabs > .tabitem {
    padding-top: 1rem !important;
}

/* ── Chat section separator ────────────────────────────────────────────── */
#follow-up-chatbot {
    border-top: 1px solid #E8E9F3 !important;
    padding-top: 1rem !important;
    margin-top: 0.5rem !important;
}

/* ── Mobile responsive ─────────────────────────────────────────────────── */
@media (max-width: 768px) {
    #main-tabs, #hp-footer {
        width: 100% !important;
        max-width: 100% !important;
    }
    .gradio-container {
        padding: 0 0.5rem !important;
    }
    #hp-header {
        padding: 2rem 1rem !important;
    }
    #hp-header h1 {
        font-size: 1.75rem !important;
    }
    #hp-header p {
        font-size: 0.875rem !important;
    }
    /* Stack rows vertically on mobile */
    #attachments-row {
        flex-direction: column !important;
    }
    #attachments-row > div {
        width: 100% !important;
        max-width: 100% !important;
    }
    .gr-row {
        flex-wrap: wrap !important;
    }
    .gr-group {
        padding: 0.75rem !important;
        border-radius: 12px !important;
    }
    #run-workflow-btn {
        height: 48px !important;
        font-size: 0.9rem !important;
    }
    .tabs > .tab-nav {
        flex-wrap: wrap !important;
    }
    .tabs > .tab-nav > button {
        padding: 6px 12px !important;
        font-size: 0.85rem !important;
    }
}
"""


def create_interface() -> gr.Blocks:
    """Build and return the Gradio ``Blocks`` application.

    Returns:
        Configured ``gr.Blocks`` instance ready for ``.launch()``.
    """
    with gr.Blocks(
        title="HealthPost \u2014 AI Clinical Decision Support",
        theme=theme,
        css=CUSTOM_CSS,
        head=CUSTOM_HEAD,
        fill_width=True,
        js="""() => {
            const fix = () => {
                const hdr = document.getElementById('hp-header-wrap');
                if (!hdr) return false;
                let p = hdr.parentElement;
                while (p) {
                    p.style.setProperty('width', '100%', 'important');
                    p.style.setProperty('max-width', '100%', 'important');
                    p.style.setProperty('padding-left', '0', 'important');
                    p.style.setProperty('padding-right', '0', 'important');
                    if (p.classList.contains('gradio-container')) break;
                    p = p.parentElement;
                }
                return true;
            };
            if (!fix()) {
                new MutationObserver((_, obs) => {
                    if (fix()) obs.disconnect();
                }).observe(document.body, {childList: true, subtree: true});
            }
            setTimeout(fix, 500);
            setTimeout(fix, 2000);
        }""",
    ) as app:

        # Header
        gr.HTML(HEADER_HTML, elem_id="hp-header-wrap")

        with gr.Tabs(elem_id="main-tabs"):

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
                    gr.HTML('<p style="font-size:1.15rem; font-weight:600; margin:0.25rem 0 0.75rem; color:#1A1F2E; font-family:Inter,sans-serif;"><span style="color:#0066FF; margin-right:0.5rem;">●</span>Patient Information</p>')
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
                with gr.Row(elem_id="attachments-row"):
                    with gr.Column():
                        with gr.Accordion(
                            "Add Medical Image (optional)", open=False,
                        ):
                            quick_image = gr.Image(
                                label="Upload a medical photo",
                                type="numpy",
                            )
                    with gr.Column():
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
                    elem_id="run-workflow-btn",
                )

                # Diagnostic output
                quick_output = gr.Markdown(
                    label="Diagnostic Report",
                    elem_id="diagnostic-output",
                )

                # --- Post-diagnosis chat ---
                visit_result_state = gr.State(None)
                chat_messages_state = gr.State([])

                chat_header = gr.HTML(
                    '<p style="font-size:1.15rem; font-weight:600; margin:0.25rem 0 0.5rem; color:#1A1F2E; font-family:Inter,sans-serif;">'
                    '<span style="color:#0066FF; margin-right:0.5rem;">●</span>Follow-up Questions</p>'
                    '<p style="font-size:0.9rem; color:#6B7280; margin:0 0 0.75rem; font-family:Inter,sans-serif;">'
                    'Ask questions about the diagnosis, dosage, referral criteria, etc.</p>',
                    visible=False,
                )

                chat_chatbot = gr.Chatbot(
                    label="Ask about this diagnosis",
                    visible=False,
                    height=300,
                    layout="bubble",
                    type="messages",
                    elem_id="follow-up-chatbot",
                )
                with gr.Row(visible=False) as chat_input_row:
                    chat_textbox = gr.Textbox(
                        placeholder="e.g., What if the patient is pregnant?",
                        show_label=False,
                        scale=9,
                    )
                    chat_send_btn = gr.Button(
                        "Send", variant="primary", scale=1,
                        elem_id="chat-send-btn",
                    )

                demo_dropdown.change(
                    fn=load_demo_scenario,
                    inputs=demo_dropdown,
                    outputs=[
                        quick_symptoms, quick_age,
                        quick_meds_text,
                    ],
                )

                quick_run_btn.click(
                    fn=run_complete_workflow,
                    inputs=[
                        quick_audio, quick_symptoms, quick_image,
                        quick_age,
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
              with gr.Group(elem_id="about-content"):
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
        gr.HTML(FOOTER_HTML, elem_id="hp-footer")

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("=" * 50)
    print("HealthPost - CHW Decision Support System")
    print("=" * 50)
    print("\nPre-loading models...")
    get_healthpost().warmup()
    print("Models ready!\n")

    print("Starting application...")

    application = create_interface()
    application.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=args.share,
    )
