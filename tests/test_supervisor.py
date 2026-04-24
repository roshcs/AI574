"""Tests for supervisor routing metadata and routing-only benchmarks."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from evaluation.metrics import Evaluator
from orchestration.supervisor import SupervisorAgent


def _mock_llm(routing_json: dict):
    llm = MagicMock()
    llm.invoke_and_parse_json.return_value = routing_json
    llm.invoke.return_value = SimpleNamespace(content="clarify question")
    return llm


class TestSupervisorRichRouting(unittest.TestCase):
    def test_route_returns_rich_metadata(self):
        llm = _mock_llm({
            "domain": "recipe",
            "confidence": 0.91,
            "second_domain": "scientific",
            "second_confidence": 0.22,
            "ambiguity": 0.10,
            "requires_clarification": False,
            "reasoning": "Ingredient substitution request",
        })
        supervisor = SupervisorAgent(llm)

        routing = supervisor.route("What can I substitute for eggs?")

        self.assertEqual(routing["domain"], "recipe")
        self.assertEqual(routing["second_domain"], "scientific")
        self.assertAlmostEqual(routing["second_confidence"], 0.22)
        self.assertAlmostEqual(routing["ambiguity"], 0.10)
        self.assertFalse(routing["requires_clarification"])

    def test_close_second_choice_requests_clarification(self):
        llm = _mock_llm({
            "domain": "recipe",
            "confidence": 0.82,
            "second_domain": "industrial",
            "second_confidence": 0.75,
            "ambiguity": 0.80,
            "requires_clarification": False,
            "reasoning": "Could be oven cooking or equipment calibration",
        })
        supervisor = SupervisorAgent(llm)

        routing = supervisor.route("How do I calibrate my oven thermometer?")

        self.assertEqual(routing["domain"], "clarify")
        self.assertTrue(routing["requires_clarification"])
        self.assertEqual(routing["second_domain"], "industrial")

    def test_fallback_routing_includes_rich_fields(self):
        routing = SupervisorAgent._fallback_routing("out of scope")

        self.assertEqual(routing["domain"], "fallback")
        self.assertEqual(routing["second_domain"], "")
        self.assertEqual(routing["second_confidence"], 0.0)
        self.assertEqual(routing["ambiguity"], 1.0)
        self.assertFalse(routing["requires_clarification"])


class TestSupervisorRoutingBenchmark(unittest.TestCase):
    def test_evaluate_supervisor_routing_computes_metrics(self):
        class FakeSupervisor:
            def route(self, query):
                if "PLC" in query:
                    return {
                        "domain": "industrial",
                        "confidence": 0.98,
                        "second_domain": "recipe",
                        "second_confidence": 0.05,
                        "ambiguity": 0.0,
                        "requires_clarification": False,
                        "reasoning": "PLC query",
                    }
                return {
                    "domain": "recipe",
                    "confidence": 0.95,
                    "second_domain": "industrial",
                    "second_confidence": 0.10,
                    "ambiguity": 0.0,
                    "requires_clarification": False,
                    "reasoning": "recipe query",
                }

        evaluator = Evaluator(workflow=None)
        result = evaluator.evaluate_supervisor_routing(
            supervisor=FakeSupervisor(),
            test_queries=[
                ("PLC fault F0003", "industrial", "easy", "PLC route"),
                ("Pancake recipe", "recipe", "easy", "recipe route"),
            ],
        )

        self.assertEqual(result["total_queries"], 2)
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["overall_accuracy"], 1.0)
        self.assertIn("macro_f1", result)
        self.assertEqual(result["confusion_matrix"]["industrial"]["industrial"], 1)


if __name__ == "__main__":
    unittest.main()
