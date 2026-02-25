"""
Supervisor Agent
================
Central router that classifies user intent and delegates to specialists.
Implements confidence-based routing with clarification fallback.

Usage:
    from orchestration.supervisor import SupervisorAgent
    supervisor = SupervisorAgent(llm=llm)
    routing = supervisor.route("My PLC is showing fault code F0003")
    # → {"domain": "industrial", "confidence": 0.98, ...}
"""

from __future__ import annotations

import logging
from typing import Dict

from langchain_core.messages import HumanMessage, SystemMessage

from config.prompts import SUPERVISOR_SYSTEM_PROMPT
from config.settings import CONFIG

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """
    Supervisor/Router Agent.

    Analyzes user queries and routes to the appropriate specialist agent.
    Uses few-shot prompting and confidence scoring.
    """

    def __init__(self, llm):
        self.llm = llm
        self.threshold = CONFIG.supervisor.routing_confidence_threshold
        self.valid_domains = set(CONFIG.supervisor.domains + ["clarify", "fallback"])

    def route(self, query: str) -> Dict:
        """
        Classify and route a user query.

        Returns:
            {
                "domain": str,         # target domain or "clarify"/"fallback"
                "confidence": float,   # 0.0 - 1.0
                "reasoning": str,      # explanation
            }
        """
        system_msg = SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)
        user_prompt = (
            f'User query: "{query}"\n\n'
            "Output the routing as a single JSON object with keys "
            '"domain", "confidence", and "reasoning". '
            "Respond with ONLY that JSON object, with no extra text."
        )

        try:
            result = self.llm.invoke_and_parse_json(
                [system_msg, HumanMessage(content=user_prompt)]
            )

            if "error" in result:
                logger.warning(
                    "Supervisor parse failed (rich prompt), raw: %s",
                    result.get("raw", "")[:120],
                )
                # Second attempt: fall back to a simpler, JSON-only prompt.
                backup = self._route_with_simple_prompt(query)
                if backup is not None:
                    return backup
                return self._fallback_routing(query)

            return self._build_routing_from_result(result, query)

        except Exception as e:
            logger.error(f"Supervisor routing failed: {e}")
            backup = self._route_with_simple_prompt(query)
            if backup is not None:
                return backup
            return self._fallback_routing(query)

    def generate_clarification(self, query: str, routing: Dict) -> str:
        """Generate a clarification question when routing is ambiguous."""
        prompt = (
            f"The user asked: \"{query}\"\n\n"
            f"This query is ambiguous — it could relate to multiple domains. "
            f"Routing analysis: {routing.get('reasoning', 'unclear intent')}\n\n"
            f"Generate a SHORT, friendly clarification question to determine "
            f"the user's intent. Ask about the specific domain they need help with."
        )

        try:
            result = self.llm.invoke([HumanMessage(content=prompt)])
            return result.content.strip()
        except Exception:
            return (
                "I want to make sure I help you with the right thing. "
                "Could you clarify whether this is about industrial equipment, "
                "cooking/recipes, or scientific research?"
            )

    def _build_routing_from_result(self, result: Dict, query: str) -> Dict:
        """Common post-processing for LLM routing JSON."""
        domain = result.get("domain", "fallback")
        if domain not in self.valid_domains:
            logger.warning(f"Invalid domain '{domain}', using fallback")
            domain = "fallback"

        # Validate and clamp confidence to [0, 1]
        try:
            confidence = float(result.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if not (0.0 <= confidence <= 1.0) or confidence != confidence:  # NaN check
            logger.warning(f"Confidence out of range ({confidence}), clamping")
            confidence = max(0.0, min(1.0, confidence)) if confidence == confidence else 0.0

        if confidence < self.threshold and domain not in ("clarify", "fallback"):
            logger.info(
                f"Low confidence ({confidence:.2f} < {self.threshold}), "
                f"switching to clarification"
            )
            domain = "clarify"

        routing = {
            "domain": domain,
            "confidence": confidence,
            "reasoning": result.get("reasoning", ""),
        }

        logger.info(
            f"Routed to '{domain}' (confidence={confidence:.2f}): "
            f"{routing['reasoning'][:60]}"
        )
        return routing

    def _route_with_simple_prompt(self, query: str) -> Dict | None:
        """
        Backup routing path with a much simpler JSON-only prompt.

        Used when the rich supervisor prompt fails to produce valid JSON.
        """
        prompt = (
            "You are a router for a multi-domain assistant.\n"
            "Given a user query, choose exactly one domain from:\n"
            '  - "industrial"\n'
            '  - "recipe"\n'
            '  - "scientific"\n'
            '  - "clarify"\n'
            '  - "fallback"\n\n'
            "Respond ONLY with valid JSON in this format:\n"
            '{\n'
            '  \"domain\": \"<one of the above>\",\n'
            "  \"confidence\": <0.0-1.0>,\n"
            '  \"reasoning\": \"<brief explanation>\"\n'
            "}\n\n"
            f'User query: \"{query}\"'
        )

        try:
            result = self.llm.invoke_and_parse_json([HumanMessage(content=prompt)])
            if "error" in result:
                logger.warning(
                    "Simple supervisor prompt parse failed, raw: %s",
                    result.get("raw", "")[:120],
                )
                return None
            return self._build_routing_from_result(result, query)
        except Exception as e:
            logger.error(f"Simple supervisor routing failed: {e}")
            return None

    @staticmethod
    def _fallback_routing(query: str) -> Dict:
        """Fallback when LLM routing fails entirely."""
        return {
            "domain": "fallback",
            "confidence": 0.0,
            "reasoning": "Routing failed — using web search fallback",
        }
