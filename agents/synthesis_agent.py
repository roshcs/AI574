"""
Cross-Domain Synthesis Agent
============================
Specialist agent for queries that genuinely span multiple domains.

It dispatches the user's query to several domain-specific CRAG runs, then asks
the LLM to fuse the per-domain answers into a single grounded response with
explicit per-domain attribution.

Design notes:
- Reuses the shared ``CRAGPipeline`` so each per-domain run still benefits from
  retrieval, document grading, and (optionally) hallucination checking.
- Per-domain runs default to a smaller ``k`` and skip the per-domain
  hallucination check to keep total latency manageable; the synthesis step is
  the place where grounding is reasoned over jointly.
- Returns a richer payload than ``CRAGResult`` (per-domain breakdown,
  per-domain timings) so the workflow / notebook demos can inspect what the
  synthesis actually combined.

Usage:
    synth = SynthesisAgent(crag_pipeline, llm)
    result = synth.handle(
        query="Is the keto diet supported by recent research?",
        domains=["recipe", "scientific"],
    )
    print(result["response"])
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from config.prompts import SYNTHESIS_SYSTEM_PROMPT
from rag_core.crag_pipeline import CRAGPipeline, CRAGResult

logger = logging.getLogger(__name__)


class SynthesisAgent:
    """Multi-domain synthesis specialist.

    Unlike the per-domain agents, this one does not extend ``BaseAgent`` because
    its core operation (run CRAG on N domains, then fuse) is structurally
    different from the single-domain CRAG flow.
    """

    domain = "synthesis"
    description = (
        "Cross-domain synthesis specialist. Runs grounded retrieval against "
        "multiple knowledge bases and fuses the per-domain answers into a "
        "single response with per-domain attribution."
    )

    def __init__(
        self,
        crag_pipeline: CRAGPipeline,
        llm,
        per_domain_k: int = 3,
        per_domain_skip_hallucination: bool = True,
        max_synthesis_chars: int = 6_000,
    ):
        self.crag = crag_pipeline
        self.llm = llm
        self.per_domain_k = per_domain_k
        self.per_domain_skip_hallucination = per_domain_skip_hallucination
        self.max_synthesis_chars = max_synthesis_chars

    def handle(
        self,
        query: str,
        domains: List[str],
        k: Optional[int] = None,
        max_rewrite_attempts: Optional[int] = None,
        per_domain_skip_hallucination: Optional[bool] = None,
        **_unused: Any,
    ) -> Dict[str, Any]:
        """Run CRAG against each domain and synthesize a unified answer.

        Args:
            query: User query.
            domains: Distinct domain identifiers to dispatch to. Duplicates
                are de-duplicated while preserving order. Must be non-empty.
            k: Optional per-domain retrieval ``k`` override.
            max_rewrite_attempts: Optional CRAG rewrite-attempt cap.
            per_domain_skip_hallucination: Optional override for whether each
                per-domain run skips its own hallucination check. Defaults to
                the agent-level setting (skip = True for speed).

        Returns:
            Dict with keys:
                response, sources, domains_used, per_domain_results,
                per_domain_timings, timing, escalated, escalation_reason.
        """
        ordered_domains = self._dedupe_domains(domains)
        if not ordered_domains:
            return self._empty_result(
                query, "No domains provided for synthesis"
            )

        retrieval_k = k if k is not None else self.per_domain_k
        skip_halluc = (
            per_domain_skip_hallucination
            if per_domain_skip_hallucination is not None
            else self.per_domain_skip_hallucination
        )

        per_domain_results: Dict[str, CRAGResult] = {}
        per_domain_timings: Dict[str, float] = {}

        synth_t0 = time.perf_counter()
        for domain in ordered_domains:
            t0 = time.perf_counter()
            try:
                result = self.crag.run(
                    query=query,
                    domain=domain,
                    k=retrieval_k,
                    max_rewrite_attempts=max_rewrite_attempts,
                    skip_hallucination_check=skip_halluc,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(
                    "[synthesis] CRAG run failed for domain '%s': %s",
                    domain,
                    exc,
                )
                result = CRAGResult(
                    domain=domain,
                    response="",
                    escalated=True,
                    escalation_reason=f"CRAG failure: {exc}",
                )
            per_domain_timings[domain] = time.perf_counter() - t0
            per_domain_results[domain] = result

        usable = {
            d: r
            for d, r in per_domain_results.items()
            if r.response and not r.escalated
        }

        # Even if some per-domain runs escalated, fuse what we have.
        synthesis_text, used_for_synth = self._synthesize(
            query, per_domain_results
        )
        synthesis_elapsed = time.perf_counter() - synth_t0

        sources = self._merge_sources(per_domain_results)
        all_escalated = len(usable) == 0

        return {
            "response": synthesis_text,
            "sources": sources,
            "domains_used": list(per_domain_results.keys()),
            "domains_in_synthesis": used_for_synth,
            "per_domain_results": per_domain_results,
            "per_domain_timings": per_domain_timings,
            "timing": {
                "total_s": synthesis_elapsed,
                "per_domain_s": per_domain_timings,
            },
            "escalated": all_escalated,
            "escalation_reason": (
                "All per-domain CRAG runs failed or returned no answer"
                if all_escalated
                else ""
            ),
        }

    # ── Internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _dedupe_domains(domains: List[str]) -> List[str]:
        """Preserve order, drop empties / duplicates / non-strings."""
        seen: set = set()
        ordered: List[str] = []
        for raw in domains or []:
            if not isinstance(raw, str):
                continue
            d = raw.strip()
            if not d or d in seen:
                continue
            seen.add(d)
            ordered.append(d)
        return ordered

    def _synthesize(
        self,
        query: str,
        per_domain_results: Dict[str, CRAGResult],
    ) -> tuple[str, List[str]]:
        """Call the LLM once to fuse per-domain answers into one response."""
        perspectives_block, used = self._format_perspectives(
            per_domain_results
        )

        if not used:
            # Nothing usable to synthesize — return a clear escalation message
            # rather than calling the LLM with empty perspectives.
            reasons = [
                f"- {d}: {(r.escalation_reason or 'no response').strip()[:120]}"
                for d, r in per_domain_results.items()
            ]
            return (
                "I tried to combine answers from multiple specialists for this "
                "query, but none of them returned a usable, grounded answer. "
                "Diagnostic per-domain status:\n" + "\n".join(reasons),
                [],
            )

        prompt = SYNTHESIS_SYSTEM_PROMPT.format(
            query=query,
            perspectives=perspectives_block,
        )
        try:
            llm_result = self.llm.invoke([HumanMessage(content=prompt)])
            text = (llm_result.content or "").strip()
            if not text:
                logger.warning("Synthesis LLM returned empty response")
                text = self._fallback_concatenation(used, per_domain_results)
            return text, used
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Synthesis LLM call failed: %s", exc)
            return self._fallback_concatenation(used, per_domain_results), used

    def _format_perspectives(
        self,
        per_domain_results: Dict[str, CRAGResult],
    ) -> tuple[str, List[str]]:
        """Build a numbered, bounded perspective block for the LLM."""
        per_budget = max(
            500, self.max_synthesis_chars // max(len(per_domain_results), 1)
        )
        sections: List[str] = []
        used: List[str] = []
        for domain, result in per_domain_results.items():
            response = (result.response or "").strip()
            if not response or result.escalated:
                sections.append(
                    f"### Perspective from {domain}:\n"
                    f"(no usable answer — "
                    f"{(result.escalation_reason or 'empty response').strip()[:160]})"
                )
                continue
            used.append(domain)
            sources_summary = self._summarize_sources(result.sources, limit=3)
            body = response[:per_budget]
            section = (
                f"### Perspective from {domain} "
                f"(confidence={result.confidence:.2f}, "
                f"sources={len(result.sources)}):\n"
                f"{body}"
            )
            if sources_summary:
                section += f"\nTop sources: {sources_summary}"
            sections.append(section)
        return "\n\n---\n\n".join(sections), used

    @staticmethod
    def _summarize_sources(sources: List[Dict], limit: int = 3) -> str:
        """One-line summary of the top-N sources for the perspective block."""
        if not sources:
            return ""
        labels = []
        for s in sources[:limit]:
            label = (
                s.get("source")
                or s.get("name")
                or s.get("title")
                or s.get("parent_id")
                or "unknown"
            )
            labels.append(str(label)[:60])
        return "; ".join(labels)

    @staticmethod
    def _merge_sources(
        per_domain_results: Dict[str, CRAGResult],
    ) -> List[Dict]:
        """Return all per-domain sources annotated with their origin domain."""
        merged: List[Dict] = []
        for domain, result in per_domain_results.items():
            for src in result.sources or []:
                tagged = dict(src)
                tagged["domain"] = domain
                merged.append(tagged)
        return merged

    @staticmethod
    def _fallback_concatenation(
        used: List[str],
        per_domain_results: Dict[str, CRAGResult],
    ) -> str:
        """Deterministic fallback if LLM-based synthesis fails."""
        parts = ["**Combined per-domain answer (LLM synthesis unavailable):**"]
        for domain in used:
            parts.append(f"\n**[{domain}]**\n{per_domain_results[domain].response}")
        return "\n".join(parts)

    @staticmethod
    def _empty_result(query: str, reason: str) -> Dict[str, Any]:
        return {
            "response": (
                "Cross-domain synthesis was requested, but no candidate "
                f"domains were provided. Reason: {reason}."
            ),
            "sources": [],
            "domains_used": [],
            "domains_in_synthesis": [],
            "per_domain_results": {},
            "per_domain_timings": {},
            "timing": {"total_s": 0.0, "per_domain_s": {}},
            "escalated": True,
            "escalation_reason": reason,
        }
