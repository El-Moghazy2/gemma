"""ReAct-style medical agent powered by MedGemma.

Implements an agentic workflow where MedGemma autonomously reasons through
a patient case, deciding which tools to call and when.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .drugs import DrugDatabase
from .triage import TriageAgent
from .vision import MedicalVisionAnalyzer

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = (
    "You are a medical AI agent assisting a Community Health Worker "
    "(CHW).\nYou reason step-by-step through patient cases and decide "
    "which tools to use.\n\n"
    "Available tools:\n"
    "- analyze_skin(description): Analyze skin condition from photo "
    "findings\n"
    "- analyze_wound(description): Assess wound from photo findings\n"
    "- check_interactions(drug_list): Check drug list for interactions\n"
    "- lookup_drug(drug_name): Look up drug info (dosing, "
    "contraindications)\n"
    "- get_alternative(drug_name, reason): Suggest safer alternative "
    "medication\n\n"
    "Process:\n"
    "1. Review patient information\n"
    "2. Use [THOUGHT] to reason about what you know and what you need\n"
    "3. Use [ACTION tool_name(args)] to call a tool\n"
    "4. Review [OBSERVATION] results\n"
    "5. Repeat until you have enough information\n"
    "6. Use [FINAL_ANSWER] to provide your complete assessment\n\n"
    "Format your response EXACTLY like this:\n"
    "[THOUGHT] I need to analyze the patient's symptoms and "
    "determine...\n"
    "[ACTION check_interactions(Paracetamol, Metformin)]\n"
    "[THOUGHT] Based on the interaction check, I can now provide my "
    "assessment...\n"
    "[FINAL_ANSWER]\n"
    "DIAGNOSIS: ...\n"
    "CONFIDENCE: ...\n"
    "TREATMENT: ...\n"
    "INTERACTIONS: ...\n"
    "REFERRAL: ...\n\n"
    "Always think before acting. Be thorough but concise."
)


@dataclass
class AgentStep:
    """A single step in the agent's reasoning trace.

    Attributes:
        step_type: One of ``"thought"``, ``"action"``,
            ``"observation"``, ``"final_answer"``.
        content: Textual content of this step.
    """

    step_type: str
    content: str

    def format(self) -> str:
        """Format this step for display with an icon label.

        Returns:
            Markdown-formatted string.
        """
        icons = {
            "thought": "\U0001f9e0 Thought",
            "action": "\U0001f527 Action",
            "observation": "\U0001f441 Observation",
            "final_answer": "\u2705 Final Answer",
        }
        label = icons.get(self.step_type, self.step_type.title())
        return f"**{label}:** {self.content}"


@dataclass
class AgentResult:
    """Aggregated result from a complete agent run.

    Attributes:
        diagnosis: Primary diagnosis string.
        confidence: Confidence score between 0 and 1.
        treatment: Treatment plan text.
        interactions: Drug interaction findings text.
        referral: Referral recommendation text.
        reasoning_trace: Full list of reasoning steps.
        raw_response: Concatenated raw model output for debugging.
    """

    diagnosis: str = ""
    confidence: float = 0.7
    treatment: str = ""
    interactions: str = ""
    referral: str = ""
    reasoning_trace: List[AgentStep] = field(default_factory=list)
    raw_response: str = ""

    def format_trace(self) -> str:
        """Format the reasoning trace as a numbered Markdown list.

        Returns:
            Markdown string, or placeholder if no trace exists.
        """
        if not self.reasoning_trace:
            return "*No reasoning trace available*"
        lines = [
            f"{i}. {step.format()}"
            for i, step in enumerate(self.reasoning_trace, 1)
        ]
        return "\n\n".join(lines)


class MedicalAgent:
    """ReAct agent that reasons through patient cases using tool calls.

    Iterates through ``[THOUGHT]`` -> ``[ACTION]`` -> ``[OBSERVATION]``
    cycles until reaching a ``[FINAL_ANSWER]`` or exhausting
    ``MAX_ITERATIONS``.

    Attributes:
        MAX_ITERATIONS: Maximum reasoning loops.
        TOOL_DESCRIPTIONS: Mapping of tool names to descriptions.
    """

    MAX_ITERATIONS = 5

    TOOL_DESCRIPTIONS: Dict[str, str] = {
        "analyze_skin": "Analyze skin condition from description/findings",
        "analyze_wound": "Assess wound from description/findings",
        "check_interactions": "Check drug list for interactions",
        "lookup_drug": "Look up drug info (dosing, contraindications)",
        "get_alternative": "Suggest safer alternative medication",
    }

    def __init__(
        self,
        triage_agent: TriageAgent,
        vision_analyzer: MedicalVisionAnalyzer,
        drug_db: DrugDatabase,
    ) -> None:
        """Initialize the medical agent.

        Args:
            triage_agent: Triage agent for diagnosis generation.
            vision_analyzer: Vision analyzer for image analysis.
            drug_db: Drug database for lookups and interaction checks.
        """
        self.triage = triage_agent
        self.vision = vision_analyzer
        self.drug_db = drug_db

        from .agent_graph import build_agent_graph
        self._graph = build_agent_graph(triage_agent, vision_analyzer, drug_db)

    def run(
        self,
        symptoms: str,
        visual_findings: Optional[List[str]] = None,
        current_meds: Optional[List[str]] = None,
        patient_age: Optional[str] = None,
        images: Optional[List[Any]] = None,
    ) -> AgentResult:
        """Run the full agentic workflow for a patient case.

        Delegates to the LangGraph agent graph.

        Args:
            symptoms: Text description of symptoms.
            visual_findings: Pre-analyzed visual findings.
            current_meds: Current medication list.
            patient_age: Patient age description.
            images: Raw images (analyzed if *visual_findings* is absent).

        Returns:
            ``AgentResult`` with diagnosis, treatment, interactions,
            referral, and full reasoning trace.
        """
        state = {
            "symptoms": symptoms,
            "visual_findings": visual_findings,
            "current_meds": current_meds,
            "patient_age": patient_age,
            "images": images,
            "max_iterations": self.MAX_ITERATIONS,
            "reasoning_trace": [],
        }

        final_state = self._graph.invoke(state)

        result = AgentResult(
            diagnosis=final_state.get("diagnosis", ""),
            confidence=final_state.get("confidence", 0.7),
            treatment=final_state.get("treatment", ""),
            interactions=final_state.get("interactions", ""),
            referral=final_state.get("referral", ""),
            reasoning_trace=final_state.get("reasoning_trace", []),
            raw_response=final_state.get("raw_response", ""),
        )

        return result

    def _build_context(
        self,
        symptoms: str,
        visual_findings: Optional[List[str]],
        current_meds: Optional[List[str]],
        patient_age: Optional[str],
    ) -> str:
        """Assemble the patient context block.

        Args:
            symptoms: Symptom description.
            visual_findings: Image analysis findings.
            current_meds: Current medication names.
            patient_age: Patient age string.

        Returns:
            Formatted context string.
        """
        parts = ["PATIENT CASE:"]
        if patient_age:
            parts.append(f"Age: {patient_age}")
        parts.append(f"Symptoms: {symptoms}")

        if visual_findings:
            parts.append("Visual findings:")
            for finding in visual_findings:
                parts.append(f"  - {finding}")

        if current_meds:
            parts.append(
                f"Current medications: {', '.join(current_meds)}"
            )

        return "\n".join(parts)

    def _build_agent_prompt(self, context: str) -> str:
        """Build the full agent prompt with system instructions.

        Args:
            context: Patient context block.

        Returns:
            Complete prompt string.
        """
        return (
            f"{AGENT_SYSTEM_PROMPT}\n\n{context}\n\n"
            "Now reason through this case step by step. "
            "What do you need to know? What tools should you use?\n\n"
            "[THOUGHT]"
        )

    def _parse_response(self, response: str) -> List[AgentStep]:
        """Parse a model response into structured reasoning steps.

        Handles both ``[TAG]``-structured output from MedGemma and
        unstructured text from the rule-based fallback.

        Args:
            response: Raw model response text.

        Returns:
            List of ``AgentStep`` objects.
        """
        steps: List[AgentStep] = []

        parts = re.split(
            r"\[(THOUGHT|ACTION|FINAL_ANSWER|OBSERVATION)\]?\s*",
            response,
        )

        i = 0
        while i < len(parts):
            part = parts[i].strip()
            if part in (
                "THOUGHT", "ACTION", "FINAL_ANSWER", "OBSERVATION",
            ):
                step_type = part.lower()
                content = (
                    parts[i + 1].strip() if i + 1 < len(parts) else ""
                )
                if step_type == "action":
                    content = content.rstrip("]").strip()
                if content:
                    steps.append(AgentStep(step_type, content))
                i += 2
            else:
                i += 1

        if not steps and response.strip():
            resp_upper = response.upper()
            has_diagnosis = any(
                p in resp_upper
                for p in [
                    "DIAGNOSIS:", "PRIMARY DIAGNOSIS:", "1. DIAGNOSIS",
                ]
            )
            if has_diagnosis:
                steps.append(AgentStep("final_answer", response.strip()))
            else:
                steps.append(AgentStep("thought", response.strip()))

        return steps

    def _execute_tool(
        self,
        action_str: str,
        visual_findings: Optional[List[str]],
        current_meds: Optional[List[str]],
    ) -> str:
        """Execute a tool call and return the observation.

        Args:
            action_str: Tool invocation string, e.g.
                ``"check_interactions(Aspirin, Warfarin)"``.
            visual_findings: Current visual findings.
            current_meds: Current medication list.

        Returns:
            Observation string from the tool.
        """
        match = re.match(r"(\w+)\(([^)]*)\)", action_str.strip())
        if not match:
            parts = action_str.strip().split(None, 1)
            tool_name = parts[0] if parts else action_str.strip()
            args_str = parts[1] if len(parts) > 1 else ""
        else:
            tool_name = match.group(1)
            args_str = match.group(2)

        logger.info("Executing tool: %s(%s)", tool_name, args_str)

        try:
            if tool_name == "analyze_skin":
                return self._tool_analyze_skin(visual_findings)
            if tool_name == "analyze_wound":
                return self._tool_analyze_wound(visual_findings)
            if tool_name == "check_interactions":
                return self._tool_check_interactions(
                    args_str, current_meds,
                )
            if tool_name == "lookup_drug":
                return self._tool_lookup_drug(args_str)
            if tool_name == "get_alternative":
                return self._tool_get_alternative(args_str)
            available = ", ".join(self.TOOL_DESCRIPTIONS.keys())
            return f"Unknown tool: {tool_name}. Available: {available}"
        except Exception as e:
            logger.error("Tool execution error: %s", e)
            return f"Tool error: {e}"

    def _tool_analyze_skin(
        self, visual_findings: Optional[List[str]],
    ) -> str:
        """Return skin analysis from existing visual findings.

        Args:
            visual_findings: Pre-computed findings to summarize.

        Returns:
            Observation text.
        """
        if visual_findings:
            return (
                "Skin analysis from visual findings: "
                f"{'; '.join(visual_findings[:5])}"
            )
        return (
            "No skin image available for analysis. "
            "Proceeding with symptom-based assessment."
        )

    def _tool_analyze_wound(
        self, visual_findings: Optional[List[str]],
    ) -> str:
        """Return wound assessment from existing visual findings.

        Args:
            visual_findings: Pre-computed findings to summarize.

        Returns:
            Observation text.
        """
        if visual_findings:
            return (
                "Wound assessment from visual findings: "
                f"{'; '.join(visual_findings[:5])}"
            )
        return (
            "No wound image available for analysis. "
            "Proceeding with symptom-based assessment."
        )

    def _tool_check_interactions(
        self, args_str: str, current_meds: Optional[List[str]],
    ) -> str:
        """Check drug interactions for the given drug list.

        Args:
            args_str: Comma-separated drug names from the tool call.
            current_meds: Additional medications to include.

        Returns:
            Interaction results text.
        """
        drugs = [
            d.strip().strip("'\"")
            for d in args_str.split(",")
            if d.strip()
        ]
        if current_meds:
            drugs.extend(current_meds)
        drugs = list(set(drugs))

        if len(drugs) < 2:
            return "Need at least 2 medications to check interactions."

        interactions = self.drug_db.check_interactions(drugs)
        if not interactions:
            interactions = self.drug_db._check_interactions_local(drugs)

        if not interactions:
            return f"No interactions found between: {', '.join(drugs)}"

        results = [
            f"{i.severity.upper()}: {' + '.join(i.drugs)} \u2014 "
            f"{i.description}. {i.recommendation}"
            for i in interactions
        ]
        return "\n".join(results)

    def _tool_lookup_drug(self, args_str: str) -> str:
        """Look up drug reference information.

        Args:
            args_str: Drug name to look up.

        Returns:
            Drug information text.
        """
        drug_name = args_str.strip().strip("'\"")
        info = self.drug_db.get_drug_info(drug_name)
        if info:
            contras = (
                ", ".join(info.contraindications)
                if info.contraindications
                else "None"
            )
            doses = "; ".join(
                f"{k}: {v}" for k, v in info.common_doses.items()
            )
            return (
                f"{info.name} ({info.generic_name}) \u2014 "
                f"{info.drug_class}. "
                f"Uses: {', '.join(info.common_uses)}. "
                f"Contraindications: {contras}. "
                f"Doses: {doses}"
            )
        return f"Drug '{drug_name}' not found in local database."

    def _tool_get_alternative(self, args_str: str) -> str:
        """Suggest an alternative medication.

        Args:
            args_str: ``"drug_name, reason"`` comma-separated.

        Returns:
            Alternative suggestion text.
        """
        parts = [p.strip().strip("'\"") for p in args_str.split(",")]
        drug_name = parts[0] if parts else ""

        info = self.drug_db.get_drug_info(drug_name)
        if info:
            alternatives = self.drug_db.search_drugs(info.drug_class)
            alt_names = [
                a.name
                for a in alternatives
                if a.name.lower() != drug_name.lower()
            ]
            if alt_names:
                return (
                    f"Alternatives to {drug_name} ({info.drug_class}): "
                    f"{', '.join(alt_names[:3])}"
                )
        return (
            f"Consider consulting formulary for alternatives to "
            f"{drug_name}."
        )

    def _parse_final_answer(
        self, content: str, result: AgentResult,
    ) -> None:
        """Parse final answer text into ``AgentResult`` fields.

        Supports both structured ``DIAGNOSIS: X`` and rule-based
        ``Primary diagnosis: X`` formats.

        Args:
            content: Final answer text from the model.
            result: ``AgentResult`` to populate.
        """
        for pattern in [
            r"(?:Primary )?diagnosis:\s*(.+?)(?=\n)",
            r"DIAGNOSIS:\s*(.+?)(?=\n)",
        ]:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                diag = match.group(1).strip().rstrip(".")
                if diag:
                    result.diagnosis = diag
                    break

        match = re.search(
            r"(?:CONFIDENCE|confidence)[:\s]*(high|medium|low|\d+%?)",
            content, re.IGNORECASE,
        )
        if match:
            conf_str = match.group(1).lower()
            conf_map = {"high": 0.85, "medium": 0.7, "low": 0.5}
            if conf_str in conf_map:
                result.confidence = conf_map[conf_str]
            elif conf_str.endswith("%"):
                result.confidence = int(conf_str.rstrip("%")) / 100
        elif "medium confidence" in content.lower():
            result.confidence = 0.7
        elif "high confidence" in content.lower():
            result.confidence = 0.85
        elif "low confidence" in content.lower():
            result.confidence = 0.5

        match = re.search(
            r"(?:TREATMENT|MEDICATIONS|TREATMENT PLAN)[:\s]*"
            r"(.+?)(?=\n(?:INTERACTION|REFERRAL|WARNING|FOLLOW)|$)",
            content, re.IGNORECASE | re.DOTALL,
        )
        if match:
            result.treatment = match.group(1).strip()

        match = re.search(
            r"INTERACTIONS?:\s*(.+?)(?=\nREFERRAL|\nTREATMENT|$)",
            content, re.IGNORECASE | re.DOTALL,
        )
        if match:
            result.interactions = match.group(1).strip()

        match = re.search(
            r"REFERRAL:\s*(.+?)(?=$)",
            content, re.IGNORECASE | re.DOTALL,
        )
        if match:
            result.referral = match.group(1).strip()

