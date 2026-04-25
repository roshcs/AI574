"""Tests for the cross-domain SynthesisAgent."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from agents.synthesis_agent import SynthesisAgent
from rag_core.crag_pipeline import CRAGResult


def _llm_returning(content: str) -> MagicMock:
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content=content)
    return llm


def _crag_result(
    domain: str,
    response: str = "",
    sources: list | None = None,
    escalated: bool = False,
    confidence: float = 1.0,
    escalation_reason: str = "",
) -> CRAGResult:
    return CRAGResult(
        domain=domain,
        response=response,
        sources=sources or [],
        escalated=escalated,
        confidence=confidence,
        escalation_reason=escalation_reason,
    )


class TestSynthesisAgent(unittest.TestCase):
    def test_dispatches_to_each_domain_and_synthesizes(self):
        crag = MagicMock()
        crag.run.side_effect = [
            _crag_result(
                "recipe",
                response="Pancakes use flour, milk, eggs, baking powder.",
                sources=[{"source": "food.com/pancake", "name": "pancake"}],
            ),
            _crag_result(
                "scientific",
                response="Maillard reaction explains pancake browning.",
                sources=[{"source": "arxiv:1234", "title": "Maillard kinetics"}],
            ),
        ]
        llm = _llm_returning("Combined synthesis answer.")
        agent = SynthesisAgent(crag, llm, per_domain_k=4)

        result = agent.handle(
            "How do pancakes brown?",
            domains=["recipe", "scientific"],
        )

        self.assertEqual(crag.run.call_count, 2)
        called_domains = [call.kwargs["domain"] for call in crag.run.call_args_list]
        self.assertEqual(called_domains, ["recipe", "scientific"])
        for call in crag.run.call_args_list:
            self.assertEqual(call.kwargs["k"], 4)
            self.assertTrue(call.kwargs["skip_hallucination_check"])

        self.assertEqual(result["response"], "Combined synthesis answer.")
        self.assertEqual(result["domains_used"], ["recipe", "scientific"])
        self.assertEqual(result["domains_in_synthesis"], ["recipe", "scientific"])
        self.assertEqual(len(result["sources"]), 2)
        self.assertEqual(
            {s["domain"] for s in result["sources"]},
            {"recipe", "scientific"},
        )
        self.assertFalse(result["escalated"])

    def test_dedupes_and_skips_invalid_domains(self):
        crag = MagicMock()
        crag.run.return_value = _crag_result("recipe", response="ans")
        agent = SynthesisAgent(crag, _llm_returning("synthesis"))

        result = agent.handle(
            "q",
            domains=["recipe", "recipe", "", None, "scientific"],  # type: ignore[list-item]
        )

        called = [c.kwargs["domain"] for c in crag.run.call_args_list]
        self.assertEqual(called, ["recipe", "scientific"])
        self.assertEqual(result["domains_used"], ["recipe", "scientific"])

    def test_returns_clear_message_when_no_domains_provided(self):
        agent = SynthesisAgent(MagicMock(), _llm_returning("noop"))
        result = agent.handle("q", domains=[])

        self.assertTrue(result["escalated"])
        self.assertEqual(result["domains_used"], [])
        self.assertIn("no candidate domains", result["response"].lower())

    def test_does_not_call_llm_when_all_per_domain_runs_escalated(self):
        crag = MagicMock()
        crag.run.side_effect = [
            _crag_result("recipe", escalated=True, escalation_reason="no docs"),
            _crag_result("scientific", escalated=True, escalation_reason="no docs"),
        ]
        llm = _llm_returning("should not be used")
        agent = SynthesisAgent(crag, llm)

        result = agent.handle("q", domains=["recipe", "scientific"])

        self.assertTrue(result["escalated"])
        self.assertEqual(result["domains_in_synthesis"], [])
        llm.invoke.assert_not_called()
        self.assertIn("usable, grounded answer", result["response"].lower())
        # Diagnostic per-domain status should include both domain names and reasons.
        self.assertIn("recipe:", result["response"])
        self.assertIn("scientific:", result["response"])

    def test_falls_back_to_concatenation_if_llm_returns_empty(self):
        crag = MagicMock()
        crag.run.side_effect = [
            _crag_result("recipe", response="recipe text", sources=[{"source": "r"}]),
            _crag_result("scientific", response="science text", sources=[{"source": "s"}]),
        ]
        llm = _llm_returning("")  # blank synthesis
        agent = SynthesisAgent(crag, llm)

        result = agent.handle("q", domains=["recipe", "scientific"])

        self.assertIn("recipe text", result["response"])
        self.assertIn("science text", result["response"])
        self.assertFalse(result["escalated"])

    def test_partial_failure_still_synthesizes_usable_domains(self):
        crag = MagicMock()
        crag.run.side_effect = [
            _crag_result("recipe", response="recipe answer", sources=[{"source": "r"}]),
            _crag_result(
                "scientific",
                escalated=True,
                escalation_reason="no relevant docs",
            ),
        ]
        agent = SynthesisAgent(crag, _llm_returning("just recipe"))

        result = agent.handle("q", domains=["recipe", "scientific"])

        self.assertFalse(result["escalated"])
        self.assertEqual(result["domains_in_synthesis"], ["recipe"])
        # Both domains appear in per_domain_results, but only recipe was usable.
        self.assertEqual(
            set(result["per_domain_results"].keys()), {"recipe", "scientific"}
        )


if __name__ == "__main__":
    unittest.main()
