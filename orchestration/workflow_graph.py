"""
Workflow Graph
==============
LangGraph StateGraph that wires the supervisor, agents, and CRAG pipeline
into a complete executable workflow.

    START → supervisor_node → route_decision
        ├─ industrial → industrial_node → finalize
        ├─ recipe → recipe_node → finalize
        ├─ scientific → scientific_node → finalize
        ├─ clarify → clarify_node → END
        └─ fallback → fallback_node → finalize
    finalize → END

Usage:
    from orchestration.workflow_graph import build_workflow, run_query
    workflow = build_workflow(llm, vector_store, index_builder)
    result = run_query(workflow, "My PLC is showing fault F0003")
"""

from __future__ import annotations

import importlib
import logging
import time
from typing import Dict, Optional

from langgraph.graph import END, StateGraph

from orchestration.state_schema import AgentState, create_initial_state
from orchestration.supervisor import SupervisorAgent
from rag_core.crag_pipeline import CRAGPipeline
from foundation.vector_store import VectorStoreService
from ingestion.index_builder import IndexBuilder
from config.settings import CONFIG

logger = logging.getLogger(__name__)


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
        state["routing_reasoning"] = routing["reasoning"]
        state["status"] = "routing"

        return state
    return supervisor_node


def _make_agent_node(agent, domain_name: str):
    """Create a specialist agent execution node."""
    def agent_node(state: AgentState) -> AgentState:
        query = state["user_query"]
        state["status"] = "processing"

        t0 = time.perf_counter()
        result = agent.handle(query)
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


def _make_fallback_node(llm):
    """Create the web-search fallback node."""
    def fallback_node(state: AgentState) -> AgentState:
        # In a full implementation, this would use Tavily Search API
        state["final_response"] = (
            "This question falls outside my specialized domains "
            "(industrial troubleshooting, recipes, and scientific papers). "
            "I'd recommend searching the web for more information on this topic."
        )
        state["status"] = "complete"
        return state
    return fallback_node


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
    """Conditional edge: route to the appropriate agent node."""
    domain = state.get("routed_domain", "fallback")

    valid_routes = {spec.name for spec in CONFIG.supervisor.domain_registry}
    valid_routes |= {"clarify", "fallback"}
    if domain not in valid_routes:
        logger.warning(f"Unknown domain '{domain}', routing to fallback")
        return "fallback"

    return domain


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_workflow(
    llm,
    vector_store: VectorStoreService,
    index_builder: Optional[IndexBuilder] = None,
) -> StateGraph:
    """
    Build and compile the complete LangGraph workflow.

    Args:
        llm: LangChain-compatible chat model.
        vector_store: Initialized VectorStoreService.
        index_builder: Optional IndexBuilder for on-demand ArXiv fetching.

    Returns:
        Compiled LangGraph StateGraph ready for invocation.
    """
    # Initialize components
    supervisor = SupervisorAgent(llm)
    crag = CRAGPipeline(llm, vector_store)

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
    graph.add_node("fallback", _make_fallback_node(llm))
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("supervisor")

    routing_map["clarify"] = "clarify"
    routing_map["fallback"] = "fallback"
    graph.add_conditional_edges("supervisor", _route_decision, routing_map)

    # Domain agent nodes → finalize
    for spec in CONFIG.supervisor.domain_registry:
        graph.add_edge(spec.name, "finalize")
    graph.add_edge("fallback", "finalize")

    # Clarify and finalize → END
    graph.add_edge("clarify", END)
    graph.add_edge("finalize", END)

    # Compile
    compiled = graph.compile()
    logger.info("Workflow graph compiled successfully")
    return compiled


# ── Convenience Runner ────────────────────────────────────────────────────────

def run_query(workflow, query: str, history=None) -> Dict:
    """
    Run a query through the full workflow and return results.

    Returns:
        {
            "response": str,
            "domain": str,
            "confidence": float,
            "sources": list,
            "escalated": bool,
            "needs_clarification": bool,
            "timing": { "total_s", "supervisor_s", "agent_s", "crag": {...} },
        }
    """
    t0 = time.perf_counter()
    initial_state = create_initial_state(query, history)
    final_state = workflow.invoke(initial_state)
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
        "sources": final_state.get("agent_sources", []),
        "escalated": final_state.get("escalated", False),
        "needs_clarification": final_state.get("needs_clarification", False),
        "status": final_state.get("status", "unknown"),
        "timing": timing,
    }
