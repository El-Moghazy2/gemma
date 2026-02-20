"""LangGraph-based patient visit pipeline.

Replaces the linear orchestration in ``core.py`` with a ``StateGraph``
that supports conditional routing (e.g. skip image analysis when no
images, route to agentic vs linear diagnosis).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)


class VisitState(TypedDict, total=False):
    """Typed state flowing through the patient visit graph."""

    # Inputs
    audio: Optional[Any]
    symptoms_text: Optional[str]
    images: Optional[List[Any]]
    existing_meds_photo: Optional[Any]
    existing_meds_list: Optional[List[str]]
    patient_age: Optional[str]
    use_agentic: bool

    # Pipeline
    symptoms: str
    visual_findings: List[str]
    current_meds: List[str]
    diagnosis: Any  # triage.Diagnosis
    treatment_plan: Any  # triage.TreatmentPlan
    drug_interactions: list
    alternative_medications: Dict[str, str]
    is_safe_to_proceed: bool
    needs_referral: bool
    referral_reason: Optional[str]
    overall_confidence: float
    reasoning_trace: List[str]

    # Agentic result (intermediate)
    agent_result: Any


def build_visit_graph(
    hp,
    on_progress: Optional[Callable[[str, str], None]] = None,
    on_agent_step: Optional[Callable] = None,
) -> Any:
    """Build and compile the patient visit pipeline graph.

    Args:
        hp: ``HealthPost`` instance whose subsystems are used by nodes.
        on_progress: Optional callback ``(step_name, description)``
            invoked at the start of each pipeline node.
        on_agent_step: Optional callback for agent reasoning steps,
            passed through to ``MedicalAgent.run()``.

    Returns:
        Compiled LangGraph ``StateGraph``.
    """

    def _notify(step_name: str, description: str) -> None:
        logger.info("Pipeline step: %s - %s", step_name, description)
        if on_progress:
            on_progress(step_name, description)

    def intake(state: VisitState) -> dict:
        """Transcribe audio or use text symptoms."""
        _notify("intake", "Capturing symptoms...")
        audio = state.get("audio")
        symptoms_text = state.get("symptoms_text")

        if audio is not None:
            symptoms = hp.voice.transcribe(audio)
        elif symptoms_text:
            symptoms = symptoms_text
        else:
            symptoms = "No symptoms provided"

        logger.info("Symptoms captured: %s...", symptoms[:100])

        return {
            "symptoms": symptoms,
            "visual_findings": [],
            "current_meds": [],
            "drug_interactions": [],
            "alternative_medications": {},
            "reasoning_trace": [],
        }

    def analyze_images(state: VisitState) -> dict:
        """Run vision analysis on each provided image."""
        _notify("analyze_images", "Analyzing medical images...")
        images = state.get("images", [])
        visual_findings: List[str] = []

        for img in images:
            try:
                findings = hp.vision.analyze_medical_image(img)
                visual_findings.extend(findings)
            except Exception as e:
                logger.warning("Image analysis failed: %s", e)

        logger.info("Visual findings: %d items", len(visual_findings))
        return {"visual_findings": visual_findings}

    def extract_meds(state: VisitState) -> dict:
        """Extract medications from photo and merge with text list."""
        _notify("extract_meds", "Extracting medications...")
        existing_meds_photo = state.get("existing_meds_photo")
        existing_meds_list = state.get("existing_meds_list")

        current_meds: List[str] = []
        if existing_meds_photo is not None:
            try:
                current_meds = hp.vision.extract_medications(
                    existing_meds_photo,
                )
            except Exception as e:
                logger.warning("Medication extraction failed: %s", e)
        if existing_meds_list:
            current_meds.extend(existing_meds_list)
        current_meds = list(set(current_meds))

        logger.info("Current medications: %s", current_meds)
        return {"current_meds": current_meds}

    def diagnose_linear(state: VisitState) -> dict:
        """Run the standard triage diagnosis pipeline."""
        _notify("diagnose", "Generating diagnosis...")
        diagnosis, treatment = hp.triage.diagnose_and_treat(
            symptoms=state["symptoms"],
            visual_findings=state.get("visual_findings", []),
        )
        logger.info(
            "Diagnosis: %s (%.0f%%)",
            diagnosis.condition, diagnosis.confidence * 100,
        )
        return {
            "diagnosis": diagnosis,
            "treatment_plan": treatment,
        }

    def diagnose_agentic(state: VisitState) -> dict:
        """Run the agentic ReAct diagnosis workflow."""
        _notify("diagnose", "Starting agentic reasoning...")
        from .agent import AgentResult
        from .triage import Diagnosis, Medication, TreatmentPlan

        agent_result = hp.agent.run(
            symptoms=state["symptoms"],
            current_meds=state.get("current_meds"),
            patient_age=state.get("patient_age"),
            images=state.get("images"),
            on_step=on_agent_step,
        )

        diagnosis = Diagnosis(
            condition=agent_result.diagnosis or "Undetermined",
            confidence=agent_result.confidence,
            supporting_evidence=[
                f"Symptoms: {state['symptoms'][:100]}",
            ],
        )

        medications = hp._parse_agent_medications(agent_result)
        needs_referral = "yes" in (agent_result.referral or "").lower()
        referral_reason = agent_result.referral if needs_referral else None

        treatment = TreatmentPlan(
            medications=medications,
            requires_referral=needs_referral,
            referral_reason=referral_reason,
        )

        trace_strings = [
            step.format() for step in agent_result.reasoning_trace
        ]

        return {
            "diagnosis": diagnosis,
            "treatment_plan": treatment,
            "reasoning_trace": trace_strings,
            "agent_result": agent_result,
        }

    def check_drugs(state: VisitState) -> dict:
        """Check drug interactions between current and proposed meds."""
        _notify("check_drugs", "Checking drug interactions...")
        current_meds = state.get("current_meds", [])
        treatment = state.get("treatment_plan")
        proposed_meds = (
            [m.name for m in treatment.medications] if treatment else []
        )
        all_meds = current_meds + proposed_meds

        interactions = hp.drug_db.check_interactions(all_meds)

        severe_interactions = [
            i for i in interactions if i.severity == "severe"
        ]
        is_safe = len(severe_interactions) == 0

        logger.info(
            "Interactions found: %d, Safe: %s",
            len(interactions), is_safe,
        )

        return {
            "drug_interactions": interactions,
            "is_safe_to_proceed": is_safe,
        }

    def find_alternatives(state: VisitState) -> dict:
        """Suggest alternative medications for those with interactions."""
        _notify("alternatives", "Finding alternative medications...")
        interactions = state.get("drug_interactions", [])
        diagnosis = state.get("diagnosis")
        treatment = state.get("treatment_plan")
        current_meds = state.get("current_meds", [])

        alternatives = hp._get_alternative_medications(
            diagnosis, treatment, interactions, current_meds,
        )
        return {"alternative_medications": alternatives}

    def assess_safety(state: VisitState) -> dict:
        """Determine referral need and overall confidence."""
        _notify("safety", "Assessing safety and referral needs...")
        diagnosis = state.get("diagnosis")
        treatment = state.get("treatment_plan")
        interactions = state.get("drug_interactions", [])
        visual_findings = state.get("visual_findings", [])

        needs_referral, referral_reason = hp._check_referral_needed(
            diagnosis, treatment, interactions,
        )
        overall_confidence = hp._calculate_confidence(
            diagnosis, visual_findings, interactions,
        )

        return {
            "needs_referral": needs_referral,
            "referral_reason": referral_reason,
            "overall_confidence": overall_confidence,
        }

    # --- Routing functions ---

    def route_after_intake(state: VisitState) -> str:
        """Skip image analysis if no images provided."""
        images = state.get("images")
        if images:
            return "analyze_images"
        return "extract_meds"

    def route_diagnosis_mode(state: VisitState) -> str:
        """Route to agentic or linear diagnosis."""
        if state.get("use_agentic", False):
            return "diagnose_agentic"
        return "diagnose_linear"

    def route_after_drugs(state: VisitState) -> str:
        """Skip find_alternatives in agentic mode or no interactions."""
        if state.get("use_agentic", False):
            return "assess_safety"
        interactions = state.get("drug_interactions", [])
        if interactions:
            return "find_alternatives"
        return "assess_safety"

    # --- Build graph ---

    graph = StateGraph(VisitState)

    graph.add_node("intake", intake)
    graph.add_node("analyze_images", analyze_images)
    graph.add_node("extract_meds", extract_meds)
    graph.add_node("diagnose_linear", diagnose_linear)
    graph.add_node("diagnose_agentic", diagnose_agentic)
    graph.add_node("check_drugs", check_drugs)
    graph.add_node("find_alternatives", find_alternatives)
    graph.add_node("assess_safety", assess_safety)

    graph.set_entry_point("intake")

    graph.add_conditional_edges(
        "intake",
        route_after_intake,
        {
            "analyze_images": "analyze_images",
            "extract_meds": "extract_meds",
        },
    )

    graph.add_edge("analyze_images", "extract_meds")

    graph.add_conditional_edges(
        "extract_meds",
        route_diagnosis_mode,
        {
            "diagnose_agentic": "diagnose_agentic",
            "diagnose_linear": "diagnose_linear",
        },
    )

    graph.add_edge("diagnose_linear", "check_drugs")
    graph.add_edge("diagnose_agentic", "check_drugs")

    graph.add_conditional_edges(
        "check_drugs",
        route_after_drugs,
        {
            "find_alternatives": "find_alternatives",
            "assess_safety": "assess_safety",
        },
    )

    graph.add_edge("find_alternatives", "assess_safety")
    graph.add_edge("assess_safety", END)

    return graph.compile()
