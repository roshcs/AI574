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
    routed_domain: str                       # "industrial" | "recipe" | "scientific" | "clarify" | "fallback"
    routing_confidence: float                # 0.0 - 1.0
    routing_reasoning: str                   # Why this domain was chosen

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


def create_initial_state(query: str, history: Optional[List[BaseMessage]] = None) -> AgentState:
    """Create a fresh state for a new query."""
    return AgentState(
        user_query=query,
        conversation_history=history or [],
        routed_domain="",
        routing_confidence=0.0,
        routing_reasoning="",
        retrieved_documents=[],
        relevant_documents=[],
        current_query=query,
        rewrite_count=0,
        agent_response="",
        agent_sources=[],
        agent_confidence=0.0,
        agent_grounded=True,
        escalated=False,
        escalation_reason="",
        needs_clarification=False,
        clarification_message="",
        final_response="",
        status="routing",
    )
