"""
Web-Search Fallback Agent
=========================
Specialist agent invoked when the supervisor routes a query to ``fallback`` —
i.e. the query falls outside every registered specialist domain.

Instead of refusing, this agent runs a live web search, then asks the LLM to
answer the query using only the retrieved snippets. This turns the previously
dead ``fallback`` route into a useful path while preserving grounded behaviour.

Search providers (selected automatically, override via ``provider=`` kwarg):
- ``tavily`` — preferred when ``TAVILY_API_KEY`` is set (better coverage).
- ``ddgs`` — DuckDuckGo via the ``duckduckgo-search`` (or newer ``ddgs``)
  package. No API key required.
- ``none`` — no provider available; the agent returns a clear, non-hallucinated
  refusal so the workflow still completes.

Usage:
    web_agent = WebSearchAgent(llm=llm)
    result = web_agent.handle("Who won the 2024 Super Bowl?")
    print(result["response"])
    for src in result["sources"]:
        print(src["url"], src["title"])
"""

from __future__ import annotations

import importlib
import logging
import os
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from config.prompts import WEB_SEARCH_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# Order tried by ``provider="auto"`` (Tavily first when key is set).
_PROVIDER_ORDER = ("tavily", "ddgs")


class WebSearchAgent:
    """Web-search-backed fallback agent."""

    domain = "fallback"
    description = (
        "Web-search fallback. Runs a live search for queries outside the "
        "registered specialist domains, then answers using only the "
        "retrieved snippets with explicit URL citations."
    )

    def __init__(
        self,
        llm,
        provider: str = "auto",
        max_results: int = 5,
        max_snippet_chars: int = 800,
        request_timeout_s: float = 8.0,
    ):
        self.llm = llm
        self.requested_provider = provider
        self.max_results = max_results
        self.max_snippet_chars = max_snippet_chars
        self.request_timeout_s = request_timeout_s
        self.provider = self._resolve_provider(provider)
        logger.info(
            "[web_search] initialized provider=%s (requested=%s)",
            self.provider,
            provider,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def handle(
        self,
        query: str,
        k: Optional[int] = None,
        **_unused: Any,
    ) -> Dict[str, Any]:
        """Search the web and answer with the LLM using only the snippets."""
        if not query or not query.strip():
            return self._refusal_result(query, "Empty query")

        target_k = k if k is not None else self.max_results

        t0 = time.perf_counter()
        try:
            results = self._search(query, target_k)
        except Exception as exc:
            logger.error("[web_search] search failed: %s: %s", type(exc).__name__, exc)
            results = []
        search_elapsed = time.perf_counter() - t0

        if not results:
            return self._refusal_result(
                query,
                reason=(
                    "Web search is unavailable in this environment "
                    f"(provider={self.provider})."
                ),
                search_elapsed=search_elapsed,
            )

        gen_t0 = time.perf_counter()
        snippets_block = self._format_snippets(results)
        prompt = WEB_SEARCH_SYSTEM_PROMPT.format(
            snippets=snippets_block,
            query=query,
        )
        try:
            llm_result = self.llm.invoke([HumanMessage(content=prompt)])
            answer = (llm_result.content or "").strip()
        except Exception as exc:
            logger.error("[web_search] LLM call failed: %s", exc)
            answer = self._fallback_answer(results)
        gen_elapsed = time.perf_counter() - gen_t0

        if not answer:
            answer = self._fallback_answer(results)

        sources = [self._result_to_source(r, idx) for idx, r in enumerate(results, 1)]

        return {
            "response": answer,
            "sources": sources,
            "provider": self.provider,
            "num_results": len(results),
            "timing": {
                "search_s": search_elapsed,
                "generate_s": gen_elapsed,
                "total_s": search_elapsed + gen_elapsed,
            },
            "escalated": False,
            "escalation_reason": "",
        }

    # ── Provider plumbing ─────────────────────────────────────────────────

    def _resolve_provider(self, provider: str) -> str:
        if provider == "none":
            return "none"
        if provider == "tavily":
            return "tavily" if self._has_tavily() else "none"
        if provider in ("ddgs", "duckduckgo"):
            return "ddgs" if self._has_ddgs() else "none"
        # auto
        for candidate in _PROVIDER_ORDER:
            if candidate == "tavily" and self._has_tavily():
                return "tavily"
            if candidate == "ddgs" and self._has_ddgs():
                return "ddgs"
        return "none"

    @staticmethod
    def _has_tavily() -> bool:
        if not os.getenv("TAVILY_API_KEY"):
            return False
        try:
            importlib.import_module("tavily")
            return True
        except ImportError:
            logger.debug(
                "[web_search] TAVILY_API_KEY set but 'tavily' package not installed"
            )
            return False

    @staticmethod
    def _has_ddgs() -> bool:
        for module_name in ("ddgs", "duckduckgo_search"):
            try:
                importlib.import_module(module_name)
                return True
            except ImportError:
                continue
        return False

    def _search(self, query: str, k: int) -> List[Dict]:
        if self.provider == "tavily":
            return self._search_tavily(query, k)
        if self.provider == "ddgs":
            return self._search_ddgs(query, k)
        return []

    def _search_tavily(self, query: str, k: int) -> List[Dict]:
        tavily = importlib.import_module("tavily")
        client = tavily.TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        try:
            raw = client.search(
                query=query,
                max_results=k,
                search_depth="basic",
            )
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("[web_search] tavily call failed: %s", exc)
            return []

        results = []
        for r in raw.get("results", []) or []:
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            content = (r.get("content") or r.get("snippet") or "").strip()
            if not (title or content):
                continue
            results.append({"title": title, "url": url, "snippet": content})
        return results[:k]

    def _search_ddgs(self, query: str, k: int) -> List[Dict]:
        # Newer package name first; fall back to legacy.
        ddgs_module = None
        for module_name in ("ddgs", "duckduckgo_search"):
            try:
                ddgs_module = importlib.import_module(module_name)
                break
            except ImportError:
                continue
        if ddgs_module is None:
            return []

        ddgs_cls = getattr(ddgs_module, "DDGS", None)
        if ddgs_cls is None:
            logger.warning("[web_search] DDGS class not found in %s", ddgs_module)
            return []

        results: List[Dict] = []
        try:
            with ddgs_cls() as ddgs:
                for r in ddgs.text(query, max_results=k):
                    title = (r.get("title") or "").strip()
                    url = (r.get("href") or r.get("url") or "").strip()
                    snippet = (r.get("body") or r.get("snippet") or "").strip()
                    if not (title or snippet):
                        continue
                    results.append({"title": title, "url": url, "snippet": snippet})
                    if len(results) >= k:
                        break
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("[web_search] ddgs call failed: %s", exc)
            return results
        return results

    # ── Formatting helpers ────────────────────────────────────────────────

    def _format_snippets(self, results: List[Dict]) -> str:
        parts = []
        for i, r in enumerate(results, 1):
            title = r.get("title") or "(no title)"
            url = r.get("url") or "(no url)"
            snippet = (r.get("snippet") or "")[: self.max_snippet_chars]
            parts.append(f"[{i}] {title}\nURL: {url}\n{snippet}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _result_to_source(result: Dict, idx: int) -> Dict:
        return {
            "rank": idx,
            "source": result.get("url", ""),
            "url": result.get("url", ""),
            "title": result.get("title", ""),
            "snippet": (result.get("snippet") or "")[:300],
        }

    @staticmethod
    def _fallback_answer(results: List[Dict]) -> str:
        """Deterministic answer if the LLM call fails after a successful search."""
        lines = [
            "I retrieved web results for your query but the language model "
            "could not synthesize them. Top sources:",
        ]
        for i, r in enumerate(results[:5], 1):
            lines.append(
                f"  [{i}] {r.get('title') or '(no title)'} — {r.get('url') or ''}"
            )
        return "\n".join(lines)

    def _refusal_result(
        self,
        query: str,
        reason: str,
        search_elapsed: float = 0.0,
    ) -> Dict[str, Any]:
        return {
            "response": (
                "This question is outside my specialist domains "
                "(industrial troubleshooting, recipes, scientific papers), "
                f"and I could not run a web search ({reason}). "
                "Please try a more specific query within one of those domains "
                "or search the web directly."
            ),
            "sources": [],
            "provider": self.provider,
            "num_results": 0,
            "timing": {
                "search_s": search_elapsed,
                "generate_s": 0.0,
                "total_s": search_elapsed,
            },
            "escalated": True,
            "escalation_reason": reason,
        }
