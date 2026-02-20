"""LangGraph ReAct agent powered by MedGemma.

Uses @tool-decorated functions and ToolNode for proper tool dispatch.
Replaces the hand-rolled ReAct loop with 2 registered tools:
  - analyze_image: vision analysis on patient photos
  - check_drug_interactions: drug DB safety queries
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode

from .drugs import DrugDatabase
from .vision import MedicalVisionAnalyzer

logger = logging.getLogger(__name__)

# ── System prompt template ──────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a medical AI agent assisting a Community Health Worker (CHW).
You reason step-by-step through patient cases and decide which tools to use.

Available tools:
{tools}

Process:
1. Review patient information
2. Use [THOUGHT] to reason about what you know and what you need
3. Use [ACTION tool_name(args)] to call a tool
4. Review [OBSERVATION] results from tool calls
5. Repeat until you have enough information
6. Use [FINAL_ANSWER] to provide your complete assessment

Format your response EXACTLY like this:
[THOUGHT] I need to analyze the patient's symptoms and determine...
[ACTION check_drug_interactions(Paracetamol, Metformin)]
[THOUGHT] Based on the results, I can now provide my assessment...
[FINAL_ANSWER]
DIAGNOSIS: ...
CONFIDENCE: ...
TREATMENT: ...
INTERACTIONS: ...
REFERRAL: ...

Important:
- Only use tools listed above. Do NOT invent tools.
- If no tool is needed, go directly to [FINAL_ANSWER].
- Always think before acting. Be thorough but concise.
- NEVER ask follow-up questions. Work with the information provided and make your best clinical assessment."""


# ── Data classes ────────────────────────────────────────────────────


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


# ── Tool factory ────────────────────────────────────────────────────


def create_tools(drug_db, vision, images=None):
    """Create @tool functions with dependency injection via closures.

    Args:
        drug_db: Drug database for interaction checks.
        vision: Vision analyzer for image analysis.
        images: Optional list of patient images.

    Returns:
        List of LangChain tool objects.
    """

    @tool
    def analyze_image(description: str) -> str:
        """Analyze the patient's medical image for clinical findings."""
        if images:
            findings = vision.analyze_medical_image(images[0])
            return "Findings: " + "; ".join(findings[:5])
        return "No image available. Proceed with symptom-based assessment."

    @tool
    def check_drug_interactions(drug_list: str) -> str:
        """Check drug-drug interactions via DDInter.
        DDInter only recognises generic/international nonproprietary names, so translate any brand or local names to their generic equivalents before calling.
        Pass comma-separated names (e.g. 'acetaminophen, metformin', not 'Tylenol, Glucophage')."""
        drugs = [d.strip() for d in drug_list.split(",") if d.strip()]
        if len(drugs) < 2:
            return "Need at least 2 drugs to check."
        interactions = drug_db.check_interactions(drugs)
        if not interactions:
            return f"No interactions found between: {', '.join(drugs)}"
        return "\n".join(
            f"{i.severity.upper()}: {' + '.join(i.drugs)} \u2014 {i.description}"
            for i in interactions
        )

    tools = [check_drug_interactions]
    if images:
        tools.insert(0, analyze_image)
    return tools


# ── Helpers ─────────────────────────────────────────────────────────


def _format_messages(system_prompt: str, messages: list) -> str:
    """Convert LangChain messages to a single text prompt for MedGemma.

    Args:
        system_prompt: System instructions.
        messages: List of LangChain message objects.

    Returns:
        Single string prompt.
    """
    parts = [f"System: {system_prompt}"]
    for msg in messages:
        if isinstance(msg, HumanMessage):
            parts.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            parts.append(f"Assistant: {msg.content}")
        elif isinstance(msg, ToolMessage):
            parts.append(f"Tool result ({msg.name}): {msg.content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _parse_tool_calls(response: str, tool_names: set, tools: list) -> list:
    """Parse ``[ACTION tool_name(args)]`` from response into ToolCall dicts.

    Only parses tools that exist in *tool_names*. Unknown tools are
    silently ignored so the agent reasons without them instead of looping.

    Args:
        response: Raw model response text.
        tool_names: Set of valid tool names.
        tools: List of tool objects (used to map arg names).

    Returns:
        List of ToolCall dicts with ``name``, ``args``, ``id``.
    """
    # Map tool name -> first parameter name
    arg_names = {}
    for t in tools:
        keys = list(t.args.keys())
        if keys:
            arg_names[t.name] = keys[0]

    tool_calls = []
    for match in re.finditer(r"\[ACTION\s+(\w+)\(([^)]*)\)\]", response):
        name = match.group(1)
        args_text = match.group(2).strip()
        if name in tool_names:
            param = arg_names.get(name, "input")
            tool_calls.append({
                "name": name,
                "args": {param: args_text},
                "id": f"call_{uuid.uuid4().hex[:8]}",
            })
            break  # Only parse the first valid tool call
    return tool_calls


def _parse_final_answer(content: str, result: AgentResult) -> None:
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


# ── Graph builder ───────────────────────────────────────────────────


def build_agent_graph(
    backend,
    config,
    tools,
    on_step: Optional[Callable[["AgentStep"], None]] = None,
):
    """Build and compile the LangGraph ReAct agent graph.

    Args:
        backend: Inference backend with ``generate_text`` method.
        config: Application config (provides ``temperature``).
        tools: List of LangChain tool objects.
        on_step: Optional callback invoked for each reasoning step.

    Returns:
        Compiled LangGraph runnable.
    """
    tool_node = ToolNode(tools)
    tool_names = {t.name for t in tools}

    # Auto-generate tool descriptions from @tool metadata
    tool_desc = "\n".join(
        f"- {t.name}({', '.join(t.args.keys())}): {t.description}"
        for t in tools
    )
    system_prompt = SYSTEM_PROMPT.format(tools=tool_desc)

    # Shared set to deduplicate progressive on_step callbacks
    _seen_steps: set = set()

    def _emit_step(step: AgentStep) -> None:
        """Emit a step via on_step callback, skipping duplicates."""
        key = (step.step_type, step.content)
        if key in _seen_steps:
            return
        _seen_steps.add(key)
        logger.info(
            "Agent step [%s]: %s",
            step.step_type, step.content[:100],
        )
        on_step(step)

    def agent(state: MessagesState):
        prompt = _format_messages(system_prompt, state["messages"])
        response = backend.generate_text(
            prompt, temperature=config.temperature, max_tokens=512,
        )
        tool_calls = _parse_tool_calls(response, tool_names, tools)
        if tool_calls:
            # Truncate content at end of first [ACTION ...] tag to prevent
            # hallucinated post-action reasoning from polluting history
            match = re.search(r"\[ACTION\s+\w+\([^)]*\)\]", response)
            if match:
                truncated = response[:match.end()]
            else:
                truncated = response
            msg = AIMessage(content=truncated, tool_calls=tool_calls)
        else:
            msg = AIMessage(content=response)

        if on_step:
            steps = _parse_response_steps(msg.content)
            for step in steps:
                _emit_step(step)

        return {"messages": [msg]}

    def _tool_node_with_callback(state: MessagesState):
        result = tool_node.invoke(state)
        if on_step:
            for msg in result.get("messages", []):
                if isinstance(msg, ToolMessage):
                    step = AgentStep(
                        "observation",
                        f"{msg.name}: {msg.content}",
                    )
                    _emit_step(step)
        return result

    def should_continue(state: MessagesState):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", _tool_node_with_callback)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", END: END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


# ── Agent class ─────────────────────────────────────────────────────


class MedicalAgent:
    """LangGraph ReAct agent that reasons through patient cases.

    Builds tools and graph per-invocation so that images and other
    per-visit data are captured in tool closures.

    Attributes:
        backend: Inference backend for text generation.
        config: Application configuration.
        drug_db: Drug database for interaction checks.
        vision: Vision analyzer for image analysis.
    """

    def __init__(
        self,
        backend,
        config,
        drug_db: DrugDatabase,
        vision: MedicalVisionAnalyzer,
    ) -> None:
        self.backend = backend
        self.config = config
        self.drug_db = drug_db
        self.vision = vision

    def run(
        self,
        symptoms: str,
        current_meds: Optional[List[str]] = None,
        patient_age: Optional[str] = None,
        images: Optional[List[Any]] = None,
        on_step: Optional[Callable[["AgentStep"], None]] = None,
    ) -> AgentResult:
        """Run the LangGraph ReAct agent for a patient case.

        Args:
            symptoms: Text description of symptoms.
            current_meds: Current medication list.
            patient_age: Patient age description.
            images: Raw images for vision analysis.
            on_step: Optional callback invoked for each reasoning step.

        Returns:
            ``AgentResult`` with diagnosis, treatment, interactions,
            referral, and full reasoning trace.
        """
        tools = create_tools(self.drug_db, self.vision, images)
        graph = build_agent_graph(
            self.backend, self.config, tools, on_step=on_step,
        )

        # Build patient case text
        parts = ["PATIENT CASE:"]
        if patient_age:
            parts.append(f"Age: {patient_age}")
        parts.append(f"Symptoms: {symptoms}")
        if current_meds:
            parts.append(
                f"Current medications: {', '.join(current_meds)}"
            )
        if images:
            parts.append(
                "Medical image(s) provided "
                "\u2014 use analyze_image to examine."
            )
        case_text = "\n".join(parts)

        # Invoke graph (recursion_limit=11 allows up to 5 agent iterations)
        initial = {"messages": [HumanMessage(content=case_text)]}
        final_state = graph.invoke(initial, {"recursion_limit": 11})

        return self._parse_result(final_state["messages"])

    def _parse_result(self, messages: list) -> AgentResult:
        """Parse the message history into an ``AgentResult``.

        Args:
            messages: Final list of LangChain messages from the graph.

        Returns:
            Populated ``AgentResult``.
        """
        result = AgentResult()
        trace: List[AgentStep] = []
        raw_parts: List[str] = []
        seen: set = set()

        for msg in messages:
            if isinstance(msg, HumanMessage):
                continue
            if isinstance(msg, AIMessage):
                raw_parts.append(msg.content)
                steps = _parse_response_steps(msg.content)
                for step in steps:
                    key = (step.step_type, step.content)
                    if key not in seen:
                        seen.add(key)
                        trace.append(step)
            elif isinstance(msg, ToolMessage):
                step = AgentStep("observation", f"{msg.name}: {msg.content}")
                key = (step.step_type, step.content)
                if key not in seen:
                    seen.add(key)
                    trace.append(step)

        # Parse final answer from the last AI message
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                _parse_final_answer(msg.content, result)
                break

        result.reasoning_trace = trace
        result.raw_response = "\n".join(raw_parts)
        return result


def _parse_response_steps(response: str) -> List[AgentStep]:
    """Parse ``[THOUGHT]``/``[ACTION]``/``[FINAL_ANSWER]`` tags from text.

    Args:
        response: Raw model response.

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
        if part in ("THOUGHT", "ACTION", "FINAL_ANSWER", "OBSERVATION"):
            step_type = part.lower()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
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
            for p in ["DIAGNOSIS:", "PRIMARY DIAGNOSIS:", "1. DIAGNOSIS"]
        )
        if has_diagnosis:
            steps.append(AgentStep("final_answer", response.strip()))
        else:
            steps.append(AgentStep("thought", response.strip()))

    return steps
