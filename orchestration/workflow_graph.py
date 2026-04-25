"""
Workflow Graph
==============
LangGraph StateGraph that wires the supervisor, agents, and CRAG pipeline
into a complete executable workflow.

    START → supervisor_node → route_decision
        ├─ industrial → industrial_node → finalize
        ├─ recipe → recipe_node → finalize
        ├─ scientific → scientific_node → finalize
        ├─ synthesis → synthesis_node → finalize    (cross-domain fusion)
        ├─ clarify → clarify_node → END
        └─ fallback → web_search_node → finalize    (live web search)
    finalize → END

The synthesis route is opt-in via ``runtime_options['enable_synthesis']``;
when enabled, the router upgrades a "clarify" decision to "synthesis" if the
supervisor produced two real specialist domains as candidates.

Usage:
    from orchestration.workflow_graph import build_workflow, run_query
    workflow = build_workflow(llm, vector_store, index_builder)

    # Default uses gemini_flash (configurable via DEFAULT_MODEL_ID):
    result = run_query(workflow, "My PLC is showing fault F0003")

    # Explicit model override:
    result = run_query(workflow, "...", model_id="groq_llama")
"""

from __future__ import annotations

import importlib
import logging
import time
from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph

from orchestration.state_schema import AgentState, create_initial_state
from orchestration.supervisor import SupervisorAgent
from rag_core.crag_pipeline import CRAGPipeline
from foundation.vector_store import VectorStoreService
from ingestion.index_builder import IndexBuilder
from agents.synthesis_agent import SynthesisAgent
from agents.web_search_agent import WebSearchAgent
from config.settings import CONFIG, DEFAULT_MODEL_ID

logger = logging.getLogger(__name__)

# ── Workflow Cache (keyed by model_id) ────────────────────────────────────────
_workflow_cache: Dict[str, Any] = {}


# ── Run Profiles ──────────────────────────────────────────────────────────────
# "best_quality" uses CONFIG defaults; "fast_interactive" trades quality for speed.

RUN_PROFILES: Dict[str, Dict] = {
    "best_quality": {
        "k": None,
        "max_rewrite_attempts": None,
        "skip_hallucination_check": None,
        "generate_max_new_tokens": None,
        "enable_synthesis": None,
    },
    "fast_interactive": {
        "k": 3,
        "max_rewrite_attempts": 0,
        "skip_hallucination_check": True,
        "generate_max_new_tokens": 256,
        "enable_synthesis": None,
    },
}

# enable_synthesis is a workflow-level routing toggle, not a CRAG runtime option.
# Keep it separate so we don't accidentally pass it into CRAGPipeline.run().
_CRAG_RUNTIME_OPTION_KEYS = frozenset({
    "k", "max_rewrite_attempts", "skip_hallucination_check", "generate_max_new_tokens",
})
_RUNTIME_OPTION_KEYS = frozenset(RUN_PROFILES["best_quality"].keys())


def _resolve_run_options(mode: str, options: Optional[Dict] = None) -> Dict:
    """Merge a named profile with optional per-call overrides.

    Profile values of ``None`` mean "fall through to per-call override or
    upstream default". For boolean toggles (e.g. ``enable_synthesis``), an
    omitted/None value resolves to ``False`` so the workflow defaults stay
    safe.
    """
    if mode not in RUN_PROFILES:
        raise ValueError(
            f"Unknown run mode '{mode}'. Choose from: {sorted(RUN_PROFILES)}"
        )
    merged = dict(RUN_PROFILES[mode])
    if options:
        unknown = set(options) - _RUNTIME_OPTION_KEYS
        if unknown:
            logger.warning("Ignoring unknown runtime options: %s", unknown)
        merged.update({k: v for k, v in options.items() if k in _RUNTIME_OPTION_KEYS})
    resolved = {k: v for k, v in merged.items() if v is not None}
    # enable_synthesis is a boolean toggle: explicit absence → False.
    resolved.setdefault("enable_synthesis", False)
    return resolved


def _import_class(dotted_path: str):
    """Import a class from a dotted path like 'agents.industrial_agent.IndustrialAgent'."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


# ── Node Functions ────────────────────────────────────────────────────────────
# Each node is a function: State → State

def _make_supervisor_node(supervisor: SupervisorAgent):
    """Create the supervisor routing node."""
    def supervisor_node(state: AgentState) -> AgentState:
        query = state["user_query"]
        t0 = time.perf_counter()
        routing = supervisor.route(query)
        elapsed = time.perf_counter() - t0
        state["timing_supervisor_s"] = elapsed
        logger.info("⏱ Supervisor (route): %.1fs", elapsed)

        state["routed_domain"] = routing["domain"]
        state["routing_confidence"] = routing["confidence"]
        state["second_routed_domain"] = routing.get("second_domain", "")
        state["second_routing_confidence"] = routing.get("second_confidence", 0.0)
        state["routing_ambiguity"] = routing.get("ambiguity", 0.0)
        state["routing_requires_clarification"] = routing.get("requires_clarification", False)
        state["routing_reasoning"] = routing["reasoning"]
        state["primary_candidate_domain"] = routing.get(
            "primary_candidate_domain", routing["domain"]
        )
        state["primary_candidate_confidence"] = routing.get(
            "primary_candidate_confidence", routing["confidence"]
        )
        state["synthesis_candidate_domains"] = routing.get(
            "synthesis_candidate_domains", []
        )
        state["status"] = "routing"

        return state
    return supervisor_node


def _crag_runtime_opts(state: AgentState) -> Dict:
    """Filter the workflow runtime options down to CRAG-safe kwargs."""
    runtime_opts = state.get("runtime_options") or {}
    return {k: v for k, v in runtime_opts.items() if k in _CRAG_RUNTIME_OPTION_KEYS}


def _make_agent_node(agent, domain_name: str):
    """Create a specialist agent execution node."""
    def agent_node(state: AgentState) -> AgentState:
        query = state["user_query"]
        state["status"] = "processing"

        crag_opts = _crag_runtime_opts(state)

        t0 = time.perf_counter()
        result = agent.handle(query, **crag_opts)
        elapsed = time.perf_counter() - t0
        state["timing_agent_s"] = elapsed
        state["timing_crag_breakdown"] = getattr(result, "timing_breakdown", {})
        logger.info("⏱ Agent (%s): %.1fs", domain_name, elapsed)

        state["agent_response"] = result.response
        state["agent_sources"] = result.sources
        state["agent_confidence"] = result.confidence
        state["agent_grounded"] = result.grounded
        state["escalated"] = result.escalated
        state["escalation_reason"] = result.escalation_reason

        return state
    return agent_node


def _make_synthesis_node(synthesis_agent: SynthesisAgent):
    """Create a cross-domain synthesis node."""
    def synthesis_node(state: AgentState) -> AgentState:
        query = state["user_query"]
        state["status"] = "processing"

        candidates = state.get("synthesis_candidate_domains") or []
        if len(candidates) < 2:
            # Defensive: router should have filtered this, but if not, escalate.
            state["agent_response"] = (
                "Cross-domain synthesis was requested but the supervisor did "
                "not produce two specialist-domain candidates. Falling back to "
                "the clarification path."
            )
            state["agent_sources"] = []
            state["agent_confidence"] = 0.0
            state["agent_grounded"] = False
            state["escalated"] = True
            state["escalation_reason"] = "Insufficient synthesis candidates"
            return state

        crag_opts = _crag_runtime_opts(state)
        # Avoid k blowing up across N domains: cap with a per-domain default.
        if "k" not in crag_opts:
            crag_opts["k"] = 3

        t0 = time.perf_counter()
        result = synthesis_agent.handle(query, domains=candidates, **crag_opts)
        elapsed = time.perf_counter() - t0
        state["timing_agent_s"] = elapsed
        logger.info(
            "⏱ Agent (synthesis over %s): %.1fs", "+".join(candidates), elapsed
        )

        # Per-domain CRAG timing breakdown is stored separately.
        state["timing_crag_breakdown"] = {
            "total_s": elapsed,
            "per_domain_s": result.get("per_domain_timings", {}),
        }

        state["agent_response"] = result["response"]
        state["agent_sources"] = result["sources"]
        state["agent_confidence"] = 1.0 if not result["escalated"] else 0.0
        state["agent_grounded"] = not result["escalated"]
        state["escalated"] = result["escalated"]
        state["escalation_reason"] = result.get("escalation_reason", "")

        # Surface synthesis-specific metadata
        per_domain = result.get("per_domain_results", {}) or {}
        state["synthesis_per_domain"] = {
            domain: {
                "response": (cr.response or "")[:400],
                "confidence": cr.confidence,
                "escalated": cr.escalated,
                "escalation_reason": cr.escalation_reason,
                "num_sources": len(cr.sources or []),
            }
            for domain, cr in per_domain.items()
        }
        state["synthesis_domains_used"] = result.get("domains_in_synthesis", [])

        return state
    return synthesis_node


def _make_web_search_node(web_agent: WebSearchAgent):
    """Create the live web-search fallback node."""
    def web_search_node(state: AgentState) -> AgentState:
        query = state["user_query"]
        state["status"] = "processing"

        t0 = time.perf_counter()
        result = web_agent.handle(query)
        elapsed = time.perf_counter() - t0
        state["timing_agent_s"] = elapsed
        logger.info(
            "⏱ Agent (web_search/%s): %.1fs (results=%d)",
            web_agent.provider,
            elapsed,
            result.get("num_results", 0),
        )
        state["timing_crag_breakdown"] = {
            "total_s": elapsed,
            "search_s": result.get("timing", {}).get("search_s", 0.0),
            "generate_s": result.get("timing", {}).get("generate_s", 0.0),
        }

        state["agent_response"] = result["response"]
        state["agent_sources"] = result["sources"]
        state["agent_confidence"] = 0.6 if not result["escalated"] else 0.0
        state["agent_grounded"] = not result["escalated"]
        state["escalated"] = result["escalated"]
        state["escalation_reason"] = result.get("escalation_reason", "")
        state["web_search_provider"] = result.get("provider", web_agent.provider)
        state["web_search_num_results"] = result.get("num_results", 0)

        return state
    return web_search_node


def _make_clarify_node(supervisor: SupervisorAgent):
    """Create the clarification node."""
    def clarify_node(state: AgentState) -> AgentState:
        query = state["user_query"]
        routing = {
            "confidence": state.get("routing_confidence", 0),
            "reasoning": state.get("routing_reasoning", ""),
        }
        t0 = time.perf_counter()
        clarification = supervisor.generate_clarification(query, routing)
        state["timing_clarify_s"] = time.perf_counter() - t0
        logger.info("⏱ Clarify (LLM): %.1fs", state["timing_clarify_s"])

        state["needs_clarification"] = True
        state["clarification_message"] = clarification
        state["final_response"] = clarification
        state["status"] = "complete"

        return state
    return clarify_node


def finalize_node(state: AgentState) -> AgentState:
    """Finalize the response — use agent output or escalation message."""
    if state.get("escalated") and not state.get("agent_response"):
        state["final_response"] = (
            f"I encountered difficulty answering your question. "
            f"Reason: {state.get('escalation_reason', 'Unknown')}. "
            f"Please try rephrasing or providing more details."
        )
    else:
        response = state.get("agent_response", "")
        # Add confidence warning if needed
        if state.get("escalated"):
            response += (
                "\n\n⚠️ *Note: I'm not fully confident in this answer. "
                "Please verify the information from the referenced sources.*"
            )
        state["final_response"] = response

    state["status"] = "complete"
    return state


# ── Routing Logic ─────────────────────────────────────────────────────────────

def _route_decision(state: AgentState) -> str:
    """Conditional edge: route to the appropriate agent node.

    Adds two routing features on top of the supervisor's chosen ``domain``:

    1. **Cross-domain synthesis upgrade.** When the supervisor returned
       ``"clarify"`` because two specialist domains were tied, and the caller
       opted in via ``runtime_options['enable_synthesis']=True``, we route to
       the synthesis node instead of asking the user a clarifying question.

    2. **Live web-search fallback.** ``"fallback"`` routes to the
       ``web_search`` node, which does a real search instead of refusing.
    """
    domain = state.get("routed_domain", "fallback")

    runtime_opts = state.get("runtime_options") or {}
    enable_synthesis = bool(runtime_opts.get("enable_synthesis", False))
    candidates = state.get("synthesis_candidate_domains") or []

    if enable_synthesis and domain == "clarify" and len(candidates) >= 2:
        logger.info(
            "Routing upgraded clarify -> synthesis over %s",
            "+".join(candidates),
        )
        return "synthesis"

    valid_specialist_routes = {spec.name for spec in CONFIG.supervisor.domain_registry}
    valid_routes = valid_specialist_routes | {"clarify", "fallback"}
    if domain not in valid_routes:
        logger.warning(f"Unknown domain '{domain}', routing to web_search fallback")
        return "fallback"

    return domain


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_workflow(
    llm,
    vector_store: VectorStoreService,
    index_builder: Optional[IndexBuilder] = None,
    web_search_provider: str = "auto",
) -> StateGraph:
    """
    Build and compile the complete LangGraph workflow.

    Args:
        llm: LangChain-compatible chat model.
        vector_store: Initialized VectorStoreService.
        index_builder: Optional IndexBuilder for on-demand ArXiv fetching.
        web_search_provider: Web-search provider for the fallback node
            (``"auto"`` | ``"tavily"`` | ``"ddgs"`` | ``"none"``).

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    # Initialize components
    supervisor = SupervisorAgent(llm)
    crag = CRAGPipeline(llm, vector_store)
    synthesis_agent = SynthesisAgent(crag, llm)
    web_agent = WebSearchAgent(llm, provider=web_search_provider)

    # Build graph
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", _make_supervisor_node(supervisor))

    # Wire domain agents from config registry (config-driven extensibility)
    routing_map: Dict[str, str] = {}
    for spec in CONFIG.supervisor.domain_registry:
        agent_cls = _import_class(spec.agent_class)
        # ScientificAgent accepts an optional index_builder
        if spec.name == "scientific" and index_builder is not None:
            agent_instance = agent_cls(crag, index_builder)
        else:
            agent_instance = agent_cls(crag)
        graph.add_node(spec.name, _make_agent_node(agent_instance, spec.name))
        routing_map[spec.name] = spec.name

    graph.add_node("clarify", _make_clarify_node(supervisor))
    graph.add_node("synthesis", _make_synthesis_node(synthesis_agent))
    graph.add_node("fallback", _make_web_search_node(web_agent))
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("supervisor")

    routing_map["clarify"] = "clarify"
    routing_map["synthesis"] = "synthesis"
    routing_map["fallback"] = "fallback"
    graph.add_conditional_edges("supervisor", _route_decision, routing_map)

    # Domain agent nodes → finalize
    for spec in CONFIG.supervisor.domain_registry:
        graph.add_edge(spec.name, "finalize")
    graph.add_edge("synthesis", "finalize")
    graph.add_edge("fallback", "finalize")

    # Clarify and finalize → END
    graph.add_edge("clarify", END)
    graph.add_edge("finalize", END)

    # Compile
    compiled = graph.compile()

    # Stash references so run_query can rebuild for a different model_id
    compiled._ai574_vector_store = vector_store
    compiled._ai574_index_builder = index_builder
    compiled._ai574_web_search_provider = web_search_provider

    logger.info(
        "Workflow graph compiled successfully (web_search=%s)",
        web_agent.provider,
    )
    return compiled


# ── Workflow Resolution ────────────────────────────────────────────────────────

def _resolve_workflow(
    workflow,
    model_id: Optional[str],
) -> Any:
    """Return the workflow to invoke, building & caching if *model_id* differs.

    If *model_id* is ``None`` the original (default) workflow is returned
    unmodified.  Otherwise a workflow compiled with the requested LLM is
    retrieved from cache or built on the fly.
    """
    if model_id is None:
        return workflow

    if model_id in _workflow_cache:
        logger.info("Reusing cached workflow for model_id='%s'", model_id)
        return _workflow_cache[model_id]

    from foundation.model_registry import get_llm

    llm = get_llm(model_id)

    vector_store = getattr(workflow, "_ai574_vector_store", None)
    index_builder = getattr(workflow, "_ai574_index_builder", None)
    web_search_provider = getattr(workflow, "_ai574_web_search_provider", "auto")

    if vector_store is None:
        raise RuntimeError(
            "Cannot rebuild workflow for a different model_id because the "
            "original workflow has no attached vector_store.  Make sure "
            "build_workflow() was used to create the workflow."
        )

    new_wf = build_workflow(
        llm,
        vector_store,
        index_builder,
        web_search_provider=web_search_provider,
    )
    _workflow_cache[model_id] = new_wf
    logger.info("Built and cached workflow for model_id='%s'", model_id)
    return new_wf


# ── Convenience Runner ────────────────────────────────────────────────────────

def run_query(
    workflow,
    query: str,
    history=None,
    mode: str = "best_quality",
    options: Optional[Dict] = None,
    model_id: Optional[str] = None,
) -> Dict:
    """
    Run a query through the full workflow and return results.

    Args:
        workflow: Compiled LangGraph workflow (from ``build_workflow``).
        query: User query string.
        history: Optional conversation history.
        mode: Run profile — ``"best_quality"`` (default) or
              ``"fast_interactive"``.  See ``RUN_PROFILES``.
        options: Per-call overrides merged on top of the selected profile.
                 Supported keys: ``k``, ``max_rewrite_attempts``,
                 ``skip_hallucination_check``, ``generate_max_new_tokens``.
        model_id: Optional model identifier (e.g. ``"gemini_flash"``,
                  ``"groq_llama"``).  When omitted, defaults to
                  ``DEFAULT_MODEL_ID`` from config (``"gemini_flash"``).
                  See ``foundation.model_registry.list_models()``.

    Returns:
        {
            "response": str,
            "domain": str,
            "confidence": float,
            "sources": list,
            "escalated": bool,
            "needs_clarification": bool,
            "mode": str,
            "model_id": str | None,
            "runtime_options": dict,
            "timing": { "total_s", "supervisor_s", "agent_s", "crag": {...} },
        }
    """
    effective_model_id = model_id if model_id is not None else DEFAULT_MODEL_ID

    runtime_options = _resolve_run_options(mode, options)
    effective_wf = _resolve_workflow(workflow, effective_model_id)

    logger.info(
        "run_query mode=%s  model_id=%s (resolved=%s)  resolved_options=%s",
        mode, model_id, effective_model_id, runtime_options,
    )

    t0 = time.perf_counter()
    initial_state = create_initial_state(query, history, runtime_options=runtime_options)
    final_state = effective_wf.invoke(initial_state)
    total_s = time.perf_counter() - t0

    crag = final_state.get("timing_crag_breakdown") or {}

    timing = {
        "total_s": total_s,
        "supervisor_s": final_state.get("timing_supervisor_s", 0.0),
        "agent_s": final_state.get("timing_agent_s", 0.0),
        "clarify_s": final_state.get("timing_clarify_s", 0.0),
        "crag": {
            "retrieve_s": crag.get("retrieve_s", 0.0),
            "grade_s": crag.get("grade_s", 0.0),
            "rewrite_s": crag.get("rewrite_s", 0.0),
            "generate_s": crag.get("generate_s", 0.0),
            "validate_s": crag.get("validate_s", 0.0),
            "total_s": crag.get("total_s", 0.0),
        },
    }
    logger.info("⏱ Total query: %.1fs (%.1f min)", total_s, total_s / 60)

    return {
        "response": final_state.get("final_response", ""),
        "domain": final_state.get("routed_domain", ""),
        "confidence": final_state.get("routing_confidence", 0.0),
        "routing_confidence": final_state.get("routing_confidence", 0.0),
        "second_domain": final_state.get("second_routed_domain", ""),
        "second_confidence": final_state.get("second_routing_confidence", 0.0),
        "routing_ambiguity": final_state.get("routing_ambiguity", 0.0),
        "routing_requires_clarification": final_state.get("routing_requires_clarification", False),
        "routing_reasoning": final_state.get("routing_reasoning", ""),
        "primary_candidate_domain": final_state.get("primary_candidate_domain", ""),
        "primary_candidate_confidence": final_state.get("primary_candidate_confidence", 0.0),
        "synthesis_candidate_domains": final_state.get("synthesis_candidate_domains", []),
        "synthesis_per_domain": final_state.get("synthesis_per_domain", {}),
        "synthesis_domains_used": final_state.get("synthesis_domains_used", []),
        "web_search_provider": final_state.get("web_search_provider", ""),
        "web_search_num_results": final_state.get("web_search_num_results", 0),
        "sources": final_state.get("agent_sources", []),
        "escalated": final_state.get("escalated", False),
        "needs_clarification": final_state.get("needs_clarification", False),
        "status": final_state.get("status", "unknown"),
        "mode": mode,
        "model_id": effective_model_id,
        "runtime_options": runtime_options,
        "timing": timing,
    }
