"""Tests for the WebSearchAgent fallback."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agents.web_search_agent import WebSearchAgent


def _llm_returning(content: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content=content)
    return llm


class TestWebSearchAgent(unittest.TestCase):
    def test_returns_refusal_when_no_provider_available(self):
        agent = WebSearchAgent(_llm_returning("never called"), provider="none")

        result = agent.handle("Who won the 2024 Super Bowl?")

        self.assertTrue(result["escalated"])
        self.assertEqual(result["provider"], "none")
        self.assertEqual(result["num_results"], 0)
        self.assertIn("outside my specialist domains", result["response"])

    def test_empty_query_short_circuits(self):
        agent = WebSearchAgent(_llm_returning("noop"), provider="none")
        result = agent.handle("")
        self.assertTrue(result["escalated"])
        self.assertIn("Empty query", result["escalation_reason"])

    def test_handle_runs_search_then_llm(self):
        llm = _llm_returning("Web answer with [1] citation.")
        agent = WebSearchAgent(llm, provider="none")
        # Force the provider as if a real one was configured.
        agent.provider = "ddgs"

        with patch.object(
            agent,
            "_search",
            return_value=[
                {"title": "Result 1", "url": "https://a.example/1", "snippet": "alpha"},
                {"title": "Result 2", "url": "https://b.example/2", "snippet": "beta"},
            ],
        ):
            result = agent.handle("Some out-of-scope question", k=2)

        llm.invoke.assert_called_once()
        prompt_arg = llm.invoke.call_args[0][0][0].content
        self.assertIn("Result 1", prompt_arg)
        self.assertIn("https://a.example/1", prompt_arg)
        self.assertIn("Some out-of-scope question", prompt_arg)

        self.assertFalse(result["escalated"])
        self.assertEqual(result["num_results"], 2)
        self.assertEqual(len(result["sources"]), 2)
        self.assertEqual(result["sources"][0]["url"], "https://a.example/1")
        self.assertEqual(result["sources"][0]["rank"], 1)

    def test_search_failure_falls_back_to_refusal(self):
        agent = WebSearchAgent(_llm_returning("nope"), provider="none")
        agent.provider = "ddgs"

        with patch.object(agent, "_search", side_effect=RuntimeError("network down")):
            result = agent.handle("anything")

        self.assertTrue(result["escalated"])
        self.assertEqual(result["num_results"], 0)
        self.assertIn("Web search is unavailable", result["response"])

    def test_llm_failure_falls_back_to_deterministic_summary(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM 503")
        agent = WebSearchAgent(llm, provider="none")
        agent.provider = "ddgs"

        with patch.object(
            agent,
            "_search",
            return_value=[
                {"title": "Top hit", "url": "https://example.com", "snippet": "..."}
            ],
        ):
            result = agent.handle("question")

        self.assertFalse(result["escalated"])
        self.assertIn("Top hit", result["response"])
        self.assertIn("https://example.com", result["response"])

    def test_provider_resolution_prefers_tavily_when_key_set(self):
        llm = _llm_returning("noop")
        with patch(
            "agents.web_search_agent.WebSearchAgent._has_tavily", return_value=True
        ), patch(
            "agents.web_search_agent.WebSearchAgent._has_ddgs", return_value=True
        ):
            self.assertEqual(WebSearchAgent(llm, provider="auto").provider, "tavily")

    def test_provider_resolution_uses_ddgs_when_only_ddgs_available(self):
        llm = _llm_returning("noop")
        with patch(
            "agents.web_search_agent.WebSearchAgent._has_tavily", return_value=False
        ), patch(
            "agents.web_search_agent.WebSearchAgent._has_ddgs", return_value=True
        ):
            self.assertEqual(WebSearchAgent(llm, provider="auto").provider, "ddgs")

    def test_provider_resolution_returns_none_when_nothing_available(self):
        llm = _llm_returning("noop")
        with patch(
            "agents.web_search_agent.WebSearchAgent._has_tavily", return_value=False
        ), patch(
            "agents.web_search_agent.WebSearchAgent._has_ddgs", return_value=False
        ):
            self.assertEqual(WebSearchAgent(llm, provider="auto").provider, "none")


if __name__ == "__main__":
    unittest.main()
