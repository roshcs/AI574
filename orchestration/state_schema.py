"""
State Schema
=============
Defines the shared state that flows through the LangGraph workflow.
All agents read from and write to this state object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    """
    Shared state for the LangGraph supervisor workflow.

    This TypedDict defines every field that can flow through the graph.
    Nodes read what they need and write their outputs back.
    """

    # ── User Input ────────────────────────────────────────────────────────
    user_query: str                          # Original user query
    conversation_history: List[BaseMessage]  # Full conversation context

    # ── Routing ───────────────────────────────────────────────────────────
    routed_domain: str                       # "industrial" | "recipe" | "scientific" | "clarify" | "fallback" | "synthesis"
    routing_confidence: float                # 0.0 - 1.0
    second_routed_domain: str                # second-best route, when available
    second_routing_confidence: float         # 0.0 - 1.0
    routing_ambiguity: float                 # 0.0 - 1.0
    routing_requires_clarification: bool     # LLM/router ambiguity signal
    routing_reasoning: str                   # Why this domain was chosen
    routing_source: str                      # "hybrid_classifier" | "llm_supervisor" | "fallback"
    classifier_margin: float                 # top-vs-second margin from hybrid classifier
    classifier_scores: Dict[str, float]      # raw hybrid classifier scores
    primary_candidate_domain: str            # original primary BEFORE clarify override
    primary_candidate_confidence: float      # confidence of original primary
    synthesis_candidate_domains: List[str]   # real domains the synthesis agent should fuse

    # ── RAG State ─────────────────────────────────────────────────────────
    retrieved_documents: List[Dict]          # Raw retrieved docs
    relevant_documents: List[Dict]           # After grading
    current_query: str                       # May differ from user_query after rewriting
    rewrite_count: int                       # Number of query rewrites so far

    # ── Agent Output ──────────────────────────────────────────────────────
    agent_response: str                      # The generated response text
    agent_sources: List[Dict]                # Source attribution
    agent_confidence: float                  # Response confidence score
    agent_grounded: bool                     # Hallucination check result

    # ── Synthesis / Web-search Extras ────────────────────────────────────
    synthesis_per_domain: Dict               # per-domain CRAG breakdown when synthesis ran
    synthesis_domains_used: List[str]        # domains actually fused into the synthesis answer
    web_search_provider: str                 # "tavily" | "ddgs" | "none" when fallback ran
    web_search_num_results: int              # number of web snippets used

    # ── Control Flow ──────────────────────────────────────────────────────
    escalated: bool                          # Whether agent escalated back
    escalation_reason: str                   # Why escalation happened
    needs_clarification: bool                # Supervisor requests more info
    clarification_message: str               # What to ask the user
    final_response: str                      # The response to return to user
    status: str                              # "routing" | "processing" | "complete" | "escalated"

    # ── Timing ────────────────────────────────────────────────────────
    timing_supervisor_s: float               # Supervisor routing time
    timing_agent_s: float                    # Agent (CRAG) processing time
    timing_clarify_s: float                  # Clarification generation time
    timing_crag_breakdown: Dict              # Per-step CRAG timing

    # ── Runtime Options ───────────────────────────────────────────────
    runtime_options: Dict                    # Per-request overrides (k, skip_hallucination_check, …)


def create_initial_state(
    query: str,
    history: Optional[List[BaseMessage]] = None,
    runtime_options: Optional[Dict] = None,
) -> AgentState:
    """Create a fresh state for a new query."""
    return AgentState(
        user_query=query,
        conversation_history=history or [],
        routed_domain="",
        routing_confidence=0.0,
        second_routed_domain="",
        second_routing_confidence=0.0,
        routing_ambiguity=0.0,
        routing_requires_clarification=False,
        routing_reasoning="",
        routing_source="",
        classifier_margin=0.0,
        classifier_scores={},
        primary_candidate_domain="",
        primary_candidate_confidence=0.0,
        synthesis_candidate_domains=[],
        retrieved_documents=[],
        relevant_documents=[],
        current_query=query,
        rewrite_count=0,
        agent_response="",
        agent_sources=[],
        agent_confidence=0.0,
        agent_grounded=True,
        synthesis_per_domain={},
        synthesis_domains_used=[],
        web_search_provider="",
        web_search_num_results=0,
        escalated=False,
        escalation_reason="",
        needs_clarification=False,
        clarification_message="",
        final_response="",
        status="routing",
        runtime_options=runtime_options or {},
    )
