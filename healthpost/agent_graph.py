"""LangGraph-based ReAct agent for medical reasoning.

Replaces the hand-rolled ReAct loop with a LangGraph ``StateGraph``
that provides proper state management, conditional routing, and a
clean graph structure.
"""

import logging
import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .drugs import DrugDatabase
from .triage import TriageAgent
from .vision import MedicalVisionAnalyzer

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5


class AgentState(TypedDict, total=False):
    """Typed state flowing through the agent graph."""

    # Inputs
    symptoms: str
    visual_findings: Optional[List[str]]
    current_meds: Optional[List[str]]
    patient_age: Optional[str]
    images: Optional[List[Any]]

    # Conversation
    conversation: str
    reasoning_trace: Annotated[list, operator.add]
    raw_response: str

    # Loop control
    iteration: int
    max_iterations: int
    thought_only_count: int

    # Parsed from last response
    current_steps: List[Dict[str, str]]
    last_response: str

    # Output
    diagnosis: str
    confidence: float
    treatment: str
    interactions: str
    referral: str


def _make_nodes(
    triage: TriageAgent,
    vision: MedicalVisionAnalyzer,
    drug_db: DrugDatabase,
):
    """Build graph node functions closed over the subsystem instances.

    Args:
        triage: Triage agent for text inference.
        vision: Vision analyzer for image analysis.
        drug_db: Drug database for lookups and interaction checks.

    Returns:
        Dict mapping node names to callables.
    """
    # Import here to avoid circular deps at module level
    from .agent import (
        AgentStep,
        MedicalAgent,
    )

    # We create a lightweight agent instance to reuse its helper methods
    agent = MedicalAgent.__new__(MedicalAgent)
    agent.triage = triage
    agent.vision = vision
    agent.drug_db = drug_db

    def prepare_context(state: AgentState) -> dict:
        """Build initial prompt from patient data and analyze images."""
        symptoms = state["symptoms"]
        visual_findings = state.get("visual_findings")
        current_meds = state.get("current_meds")
        patient_age = state.get("patient_age")
        images = state.get("images")

        trace_additions = []

        context = agent._build_context(
            symptoms, visual_findings, current_meds, patient_age,
        )

        if images and not visual_findings:
            visual_findings = []
            for img in images:
                try:
                    findings = vision.analyze_medical_image(img)
                    visual_findings.extend(findings)
                    trace_additions.append(AgentStep(
                        "observation",
                        f"Image analysis: {'; '.join(findings[:3])}",
                    ))
                except Exception as e:
                    logger.warning("Image analysis failed: %s", e)

            if visual_findings:
                context += "\nVISUAL FINDINGS:\n"
                context += "\n".join(f"- {f}" for f in visual_findings)

        prompt = agent._build_agent_prompt(context)

        return {
            "conversation": prompt,
            "visual_findings": visual_findings,
            "reasoning_trace": trace_additions,
            "raw_response": "",
            "iteration": 0,
            "thought_only_count": 0,
            "diagnosis": "",
            "confidence": 0.7,
            "treatment": "",
            "interactions": "",
            "referral": "",
        }

    def reason(state: AgentState) -> dict:
        """Call the model and parse the response into steps."""
        conversation = state["conversation"]
        iteration = state.get("iteration", 0)

        logger.info(
            "Agent iteration %d/%d",
            iteration + 1, state.get("max_iterations", MAX_ITERATIONS),
        )

        response = triage._generate_response(conversation, max_tokens=1024)
        steps = agent._parse_response(response)

        current_steps = [
            {"step_type": s.step_type, "content": s.content}
            for s in steps
        ]

        return {
            "current_steps": current_steps,
            "last_response": response,
            "raw_response": state.get("raw_response", "") + response + "\n",
            "iteration": iteration + 1,
        }

    def execute_tools(state: AgentState) -> dict:
        """Execute tool calls from the parsed steps."""
        steps = state["current_steps"]
        visual_findings = state.get("visual_findings")
        current_meds = state.get("current_meds")
        conversation = state["conversation"]
        response = state["last_response"]

        trace_additions = []
        conv_addition = ""

        for step in steps:
            trace_additions.append(
                AgentStep(step["step_type"], step["content"]),
            )

            if step["step_type"] == "action":
                observation = agent._execute_tool(
                    step["content"], visual_findings, current_meds,
                )
                obs_step = AgentStep("observation", observation)
                trace_additions.append(obs_step)
                conv_addition += (
                    f"\n{response}\n[OBSERVATION] {observation}\n"
                )

        return {
            "conversation": conversation + conv_addition,
            "reasoning_trace": trace_additions,
            "thought_only_count": 0,
        }

    def handle_thought_only(state: AgentState) -> dict:
        """Handle iterations that produced only thoughts (no actions)."""
        steps = state["current_steps"]
        conversation = state["conversation"]
        response = state["last_response"]
        thought_only_count = state.get("thought_only_count", 0) + 1

        trace_additions = [
            AgentStep(s["step_type"], s["content"]) for s in steps
        ]

        conv_addition = f"\n{response}\n"
        if thought_only_count >= 2:
            conv_addition += (
                "\nYou have enough information. "
                "Please provide your [FINAL_ANSWER] now "
                "with DIAGNOSIS, CONFIDENCE, TREATMENT, "
                "INTERACTIONS, and REFERRAL.\n"
            )

        return {
            "conversation": conversation + conv_addition,
            "reasoning_trace": trace_additions,
            "thought_only_count": thought_only_count,
        }

    def finalize(state: AgentState) -> dict:
        """Parse the final answer into output fields."""
        steps = state["current_steps"]

        trace_additions = []
        final_content = ""
        for step in steps:
            trace_additions.append(
                AgentStep(step["step_type"], step["content"]),
            )
            if step["step_type"] == "final_answer":
                final_content = step["content"]

        # Use a temporary AgentResult to parse
        from .agent import AgentResult
        temp_result = AgentResult()
        agent._parse_final_answer(final_content, temp_result)

        return {
            "diagnosis": temp_result.diagnosis,
            "confidence": temp_result.confidence,
            "treatment": temp_result.treatment,
            "interactions": temp_result.interactions,
            "referral": temp_result.referral,
            "reasoning_trace": trace_additions,
        }

    return {
        "prepare_context": prepare_context,
        "reason": reason,
        "execute_tools": execute_tools,
        "handle_thought_only": handle_thought_only,
        "finalize": finalize,
    }


def _route_after_reason(state: AgentState) -> str:
    """Route based on what the model produced."""
    steps = state.get("current_steps", [])

    for step in steps:
        if step["step_type"] == "final_answer":
            return "finalize"
        if step["step_type"] == "action":
            return "execute_tools"

    return "handle_thought_only"


def _route_continue_or_end(state: AgentState) -> str:
    """Decide whether to continue reasoning or end the graph."""
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", MAX_ITERATIONS)

    if iteration < max_iter:
        return "reason"
    return END


def build_agent_graph(
    triage: TriageAgent,
    vision: MedicalVisionAnalyzer,
    drug_db: DrugDatabase,
) -> StateGraph:
    """Build and compile the agent ReAct graph.

    Args:
        triage: Triage agent for text inference.
        vision: Vision analyzer for image analysis.
        drug_db: Drug database for lookups and interaction checks.

    Returns:
        Compiled LangGraph ``StateGraph``.
    """
    nodes = _make_nodes(triage, vision, drug_db)

    graph = StateGraph(AgentState)

    graph.add_node("prepare_context", nodes["prepare_context"])
    graph.add_node("reason", nodes["reason"])
    graph.add_node("execute_tools", nodes["execute_tools"])
    graph.add_node("handle_thought_only", nodes["handle_thought_only"])
    graph.add_node("finalize", nodes["finalize"])

    graph.set_entry_point("prepare_context")
    graph.add_edge("prepare_context", "reason")

    graph.add_conditional_edges(
        "reason",
        _route_after_reason,
        {
            "finalize": "finalize",
            "execute_tools": "execute_tools",
            "handle_thought_only": "handle_thought_only",
        },
    )

    graph.add_conditional_edges(
        "execute_tools",
        _route_continue_or_end,
        {
            "reason": "reason",
            END: END,
        },
    )

    graph.add_conditional_edges(
        "handle_thought_only",
        _route_continue_or_end,
        {
            "reason": "reason",
            END: END,
        },
    )

    graph.add_edge("finalize", END)

    return graph.compile()
