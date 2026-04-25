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
from typing import Dict, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from config.prompts import SUPERVISOR_SYSTEM_PROMPT
from config.settings import CONFIG
from orchestration.hybrid_router import LexicalDomainClassifier

logger = logging.getLogger(__name__)


class SupervisorAgent:
    """
    Supervisor/Router Agent.

    Analyzes user queries and routes to the appropriate specialist agent.
    Uses few-shot prompting and confidence scoring.
    """

    def __init__(
        self,
        llm,
        hybrid_router: Optional[LexicalDomainClassifier] = None,
        enable_hybrid_router: Optional[bool] = None,
    ):
        self.llm = llm
        self.threshold = CONFIG.supervisor.routing_confidence_threshold
        self.valid_domains = set(CONFIG.supervisor.domains + ["clarify", "fallback"])
        self.ambiguity_margin = 0.15
        self.enable_hybrid_router = (
            CONFIG.supervisor.hybrid_router_enabled
            if enable_hybrid_router is None
            else enable_hybrid_router
        )
        self.hybrid_router = hybrid_router or LexicalDomainClassifier(
            confidence_threshold=CONFIG.supervisor.hybrid_router_confidence_threshold,
            margin_threshold=CONFIG.supervisor.hybrid_router_margin_threshold,
        )

    def route(self, query: str) -> Dict:
        """
        Classify and route a user query.

        Returns:
            {
                "domain": str,         # target domain or "clarify"/"fallback"
                "confidence": float,   # 0.0 - 1.0
                "second_domain": str,
                "second_confidence": float,
                "ambiguity": float,
                "requires_clarification": bool,
                "reasoning": str,      # explanation
                "primary_candidate_domain": str,         # original primary BEFORE clarify override
                "primary_candidate_confidence": float,
                "synthesis_candidate_domains": list[str],  # real domains worth synthesising
            }
        """
        if self.enable_hybrid_router:
            hybrid_routing = self._route_with_hybrid_classifier(query)
            if hybrid_routing is not None:
                return hybrid_routing

        system_msg = SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)
        user_prompt = (
            f'User query: "{query}"\n\n'
            "Output the routing as a single JSON object with keys "
            '"domain", "confidence", "second_domain", "second_confidence", '
            '"ambiguity", "requires_clarification", and "reasoning". '
            "Respond with ONLY that JSON object, with no extra text."
        )

        try:
            result = self.llm.invoke_and_parse_json(
                [system_msg, HumanMessage(content=user_prompt)]
            )

            if "error" in result:
                logger.warning("Supervisor parse failed (rich prompt)")
                # Second attempt: fall back to a simpler, JSON-only prompt.
                backup = self._route_with_simple_prompt(query)
                if backup is not None:
                    return backup
                return self._fallback_routing(query)

            return self._build_routing_from_result(result, query, source="llm_supervisor")

        except Exception as e:
            logger.error("Supervisor routing failed: %s: %s", type(e).__name__, str(e)[:80])
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

    def _route_with_hybrid_classifier(self, query: str) -> Dict | None:
        """
        Fast first-pass routing for obvious single-domain queries.

        Returns a complete routing dictionary when the classifier is confident.
        Returns None when the query should be escalated to the LLM supervisor.
        """
        prediction = self.hybrid_router.predict(query)
        if not prediction.accepted or prediction.domain not in self.valid_domains:
            logger.info("Hybrid router escalated to LLM: %s", prediction.reason)
            return None

        ambiguity = self._infer_ambiguity(
            prediction.confidence, prediction.second_confidence
        )
        routing = {
            "domain": prediction.domain,
            "confidence": prediction.confidence,
            "second_domain": prediction.second_domain,
            "second_confidence": prediction.second_confidence,
            "ambiguity": ambiguity,
            "requires_clarification": False,
            "reasoning": prediction.reason,
            "primary_candidate_domain": prediction.domain,
            "primary_candidate_confidence": prediction.confidence,
            "synthesis_candidate_domains": [],
            "routing_source": "hybrid_classifier",
            "classifier_margin": prediction.margin,
            "classifier_scores": prediction.scores,
        }
        logger.info(
            "Hybrid router accepted '%s' (confidence=%.2f, margin=%.2f)",
            prediction.domain,
            prediction.confidence,
            prediction.margin,
        )
        return routing

    def _build_routing_from_result(
        self,
        result: Dict,
        query: str,
        source: str = "llm_supervisor",
    ) -> Dict:
        """Common post-processing for LLM routing JSON."""
        domain = result.get("domain", "fallback")
        if domain not in self.valid_domains:
            logger.warning(f"Invalid domain '{domain}', using fallback")
            domain = "fallback"

        # Validate and clamp confidence to [0, 1]
        confidence = self._coerce_confidence(result.get("confidence", 0.0))

        second_domain = self._coerce_optional_domain(result.get("second_domain"))
        if second_domain == domain:
            second_domain = ""
        second_confidence = self._coerce_confidence(result.get("second_confidence", 0.0))
        ambiguity = self._coerce_confidence(
            result.get("ambiguity", self._infer_ambiguity(confidence, second_confidence))
        )
        requires_clarification = self._coerce_bool(result.get("requires_clarification", False))

        close_second_choice = (
            second_domain
            and domain not in ("clarify", "fallback")
            and (confidence - second_confidence) < self.ambiguity_margin
        )

        # Preserve the original primary domain BEFORE we possibly override it
        # to "clarify". This lets downstream consumers (e.g. the synthesis
        # agent) recover the supervisor's actual two top candidates even when
        # the routing decision was "ask the user".
        primary_candidate_domain = domain
        primary_candidate_confidence = confidence

        if (
            (confidence < self.threshold or requires_clarification or close_second_choice)
            and domain not in ("clarify", "fallback")
        ):
            logger.info(
                "Routing needs clarification "
                "(confidence=%.2f, second=%s %.2f, ambiguity=%.2f)",
                confidence,
                second_domain or "none",
                second_confidence,
                ambiguity,
            )
            domain = "clarify"
            requires_clarification = True

        synthesis_candidate_domains = self._build_synthesis_candidates(
            primary_candidate_domain, second_domain
        )

        routing = {
            "domain": domain,
            "confidence": confidence,
            "second_domain": second_domain,
            "second_confidence": second_confidence,
            "ambiguity": ambiguity,
            "requires_clarification": requires_clarification,
            "reasoning": result.get("reasoning", ""),
            "primary_candidate_domain": primary_candidate_domain,
            "primary_candidate_confidence": primary_candidate_confidence,
            "synthesis_candidate_domains": synthesis_candidate_domains,
            "routing_source": source,
            "classifier_margin": 0.0,
            "classifier_scores": {},
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
            '  \"second_domain\": \"<second-best domain or empty string>\",\n'
            "  \"second_confidence\": <0.0-1.0>,\n"
            "  \"ambiguity\": <0.0-1.0>,\n"
            "  \"requires_clarification\": <true|false>,\n"
            '  \"reasoning\": \"<brief explanation>\"\n'
            "}\n\n"
            f'User query: \"{query}\"'
        )

        try:
            result = self.llm.invoke_and_parse_json([HumanMessage(content=prompt)])
            if "error" in result:
                logger.warning("Simple supervisor prompt parse failed")
                return None
            return self._build_routing_from_result(result, query, source="llm_supervisor")
        except Exception as e:
            logger.error("Simple supervisor routing failed: %s: %s", type(e).__name__, str(e)[:80])
            return None

    @staticmethod
    def _fallback_routing(query: str) -> Dict:
        """Fallback when LLM routing fails entirely."""
        return {
            "domain": "fallback",
            "confidence": 0.0,
            "second_domain": "",
            "second_confidence": 0.0,
            "ambiguity": 1.0,
            "requires_clarification": False,
            "reasoning": "Routing failed — using web search fallback",
            "primary_candidate_domain": "fallback",
            "primary_candidate_confidence": 0.0,
            "synthesis_candidate_domains": [],
            "routing_source": "fallback",
            "classifier_margin": 0.0,
            "classifier_scores": {},
        }

    @staticmethod
    def _coerce_confidence(value) -> float:
        """Convert model-provided numeric fields to a bounded [0, 1] float."""
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not (0.0 <= confidence <= 1.0) or confidence != confidence:  # NaN check
            logger.warning("Confidence out of range (%s), clamping", confidence)
            return max(0.0, min(1.0, confidence)) if confidence == confidence else 0.0
        return confidence

    def _coerce_optional_domain(self, value) -> str:
        """Return a valid optional domain or empty string."""
        if value in (None, "", "null"):
            return ""
        domain = str(value)
        if domain not in self.valid_domains:
            logger.warning("Invalid second_domain '%s', ignoring", domain)
            return ""
        return domain

    @staticmethod
    def _coerce_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes"}
        return bool(value)

    @staticmethod
    def _infer_ambiguity(confidence: float, second_confidence: float) -> float:
        """Estimate ambiguity when the LLM does not provide one."""
        if second_confidence <= 0:
            return max(0.0, 1.0 - confidence)
        return max(0.0, min(1.0, 1.0 - abs(confidence - second_confidence)))

    @staticmethod
    def _build_synthesis_candidates(
        primary: str, second: str
    ) -> list:
        """Return ordered, distinct list of *real* domains worth synthesising.

        Excludes pseudo-routes ("clarify", "fallback") and de-dupes. The
        cross-domain synthesis agent only makes sense when both candidates
        are real specialist domains.
        """
        pseudo = {"clarify", "fallback", ""}
        out: list = []
        for cand in (primary, second):
            if cand in pseudo or cand in out:
                continue
            out.append(cand)
        return out if len(out) >= 2 else []
