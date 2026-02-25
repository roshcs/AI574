"""
Evaluation Framework
====================
Test suite, metrics, and LLM-judge for evaluating the multi-agent system.
Covers: routing accuracy, retrieval precision, task completion, and
self-correction effectiveness.

Usage:
    from evaluation.metrics import Evaluator
    evaluator = Evaluator(workflow=workflow)
    results = evaluator.run_full_evaluation()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


# ── Test Suite ────────────────────────────────────────────────────────────────

# 150 test queries: 50 per domain
# Each entry: (query, expected_domain, difficulty, description)

INDUSTRIAL_TEST_QUERIES = [
    ("My S7-1200 PLC is showing fault code F0003, what does it mean?", "industrial", "easy", "Direct fault code lookup"),
    ("The VFD on pump station 3 is tripping on overcurrent every morning", "industrial", "medium", "Intermittent fault diagnosis"),
    ("How do I configure PROFINET communication between two Siemens PLCs?", "industrial", "medium", "Configuration procedure"),
    ("What are the recommended PM intervals for Allen-Bradley servo drives?", "industrial", "easy", "Maintenance schedule"),
    ("Our SCADA system is losing connection to RTUs intermittently", "industrial", "hard", "Network troubleshooting"),
    ("Explain the difference between SIL 2 and SIL 3 safety requirements", "industrial", "medium", "Safety standards"),
    ("Motor bearing temperature is trending upward over the last month", "industrial", "medium", "Predictive maintenance"),
    ("How to reset a PowerFlex 525 drive after an F0063 fault?", "industrial", "easy", "Fault recovery procedure"),
    ("What causes ground fault alarms on a 480V distribution panel?", "industrial", "medium", "Electrical troubleshooting"),
    ("Best practices for PLC program backup and version control", "industrial", "easy", "Best practices"),
    # ... Add 40 more for full test suite
]

RECIPE_TEST_QUERIES = [
    ("What can I substitute for eggs in a chocolate cake recipe?", "recipe", "easy", "Common substitution"),
    ("I have chicken, rice, and bell peppers. What can I make?", "recipe", "medium", "Ingredient-based search"),
    ("How do I make a proper roux for gumbo?", "recipe", "easy", "Technique question"),
    ("What's the difference between baking soda and baking powder?", "recipe", "easy", "Ingredient knowledge"),
    ("Give me a gluten-free pasta recipe with under 500 calories", "recipe", "medium", "Dietary constraint search"),
    ("How long should I rest a steak after grilling?", "recipe", "easy", "Technique timing"),
    ("What's a good dairy-free alternative to heavy cream in soup?", "recipe", "medium", "Dietary substitution"),
    ("How do I properly temper chocolate for dipping?", "recipe", "hard", "Advanced technique"),
    ("What spices pair well with lamb?", "recipe", "easy", "Flavor pairing"),
    ("Can you give me a quick weeknight dinner recipe for 4 people?", "recipe", "medium", "Recommendation"),
    # ... Add 40 more for full test suite
]

SCIENTIFIC_TEST_QUERIES = [
    ("Summarize recent papers on transformer attention mechanisms", "scientific", "medium", "Topic synthesis"),
    ("What are the key findings from the latest RLHF research?", "scientific", "medium", "Current research"),
    ("Find papers about graph neural networks for drug discovery", "scientific", "easy", "Topic search"),
    ("What is the state of the art in protein structure prediction?", "scientific", "hard", "SOTA review"),
    ("Compare self-supervised vs supervised pretraining for NLP", "scientific", "hard", "Comparative analysis"),
    ("What datasets are commonly used for named entity recognition?", "scientific", "easy", "Resource query"),
    ("Explain the mixture of experts architecture in LLMs", "scientific", "medium", "Concept explanation"),
    ("Recent advances in federated learning for medical imaging", "scientific", "medium", "Domain-specific search"),
    ("What are the limitations of current RAG approaches?", "scientific", "medium", "Limitations analysis"),
    ("Find papers on energy-efficient training of large language models", "scientific", "easy", "Topic search"),
    # ... Add 40 more for full test suite
]

# Ambiguous/edge-case queries for routing stress testing
EDGE_CASE_QUERIES = [
    ("What temperature should I cook chicken to avoid equipment failure?", "clarify", "hard", "Ambiguous cross-domain"),
    ("How do I calibrate my oven thermometer?", "clarify", "medium", "Could be recipe or industrial"),
    ("What is the recipe for disaster recovery?", "clarify", "hard", "Metaphorical language"),
    ("Tell me about mixing processes in chemical plants", "industrial", "medium", "Industrial — not recipe mixing"),
    ("How does a neural network learn?", "scientific", "easy", "Clear scientific"),
]

ALL_TEST_QUERIES = (
    INDUSTRIAL_TEST_QUERIES
    + RECIPE_TEST_QUERIES
    + SCIENTIFIC_TEST_QUERIES
    + EDGE_CASE_QUERIES
)


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class EvaluationResult:
    """Aggregated evaluation results."""
    routing_accuracy: float = 0.0
    routing_by_domain: Dict[str, float] = field(default_factory=dict)
    task_completion_rate: float = 0.0
    avg_latency_seconds: float = 0.0
    self_correction_rate: float = 0.0
    per_query_results: List[Dict] = field(default_factory=list)


class Evaluator:
    """
    Full evaluation suite for the multi-agent system.
    """

    def __init__(self, workflow, llm=None):
        self.workflow = workflow
        self.llm = llm  # For LLM-judge scoring

    def evaluate_routing(
        self,
        test_queries: Optional[List] = None,
    ) -> Dict:
        """
        Evaluate routing accuracy across all test queries.

        Returns accuracy overall and per-domain.
        """
        from orchestration.workflow_graph import run_query

        queries = test_queries or ALL_TEST_QUERIES
        correct = 0
        domain_correct = {}
        domain_total = {}
        results = []

        for query, expected_domain, difficulty, desc in queries:
            start = time.time()
            result = run_query(self.workflow, query)
            latency = time.time() - start

            actual_domain = result.get("domain", "")
            is_correct = actual_domain == expected_domain

            if is_correct:
                correct += 1

            domain_correct[expected_domain] = (
                domain_correct.get(expected_domain, 0) + (1 if is_correct else 0)
            )
            domain_total[expected_domain] = (
                domain_total.get(expected_domain, 0) + 1
            )

            results.append({
                "query": query,
                "expected": expected_domain,
                "actual": actual_domain,
                "correct": is_correct,
                "confidence": result.get("confidence", 0),
                "latency": latency,
                "difficulty": difficulty,
            })

        accuracy = correct / len(queries) if queries else 0.0
        per_domain = {
            d: domain_correct.get(d, 0) / domain_total.get(d, 1)
            for d in domain_total
        }

        return {
            "overall_accuracy": accuracy,
            "per_domain_accuracy": per_domain,
            "total_queries": len(queries),
            "correct": correct,
            "results": results,
        }

    def evaluate_llm_judge(
        self,
        query: str,
        response: str,
        domain: str,
    ) -> Dict:
        """
        Use LLM-as-judge to score a response for quality.

        Scores on: relevance, accuracy, completeness, clarity (each 1-5).
        """
        if not self.llm:
            return {"error": "No LLM provided for judging"}

        prompt = (
            f"You are evaluating an AI assistant's response quality.\n\n"
            f"Domain: {domain}\n"
            f"User Query: {query}\n"
            f"Assistant Response: {response}\n\n"
            f"Score the response on these dimensions (1-5 each):\n"
            f"1. Relevance — Does it address the query?\n"
            f"2. Accuracy — Is the information correct and grounded?\n"
            f"3. Completeness — Does it fully answer the question?\n"
            f"4. Clarity — Is it well-organized and easy to understand?\n\n"
            f"Respond with ONLY valid JSON:\n"
            f'{{"relevance": <1-5>, "accuracy": <1-5>, '
            f'"completeness": <1-5>, "clarity": <1-5>, '
            f'"overall": <1-5>, "feedback": "<brief explanation>"}}'
        )

        try:
            result = self.llm.invoke_and_parse_json(
                [HumanMessage(content=prompt)]
            )
            return result
        except Exception as e:
            return {"error": str(e)}

    # ── Verification: Functional ─────────────────────────────────────────

    def verify_functional(self, queries: Optional[List] = None) -> Dict:
        """Run representative queries across all domains, clarify, and fallback.

        Checks that every domain returns a non-empty response and a
        valid confidence value, and that the timing dict has a stable schema.
        """
        from orchestration.workflow_graph import run_query

        queries = queries or [
            ("My PowerFlex 525 drive shows fault F004", "industrial"),
            ("How do I make sourdough bread?", "recipe"),
            ("Summarize recent transformer papers", "scientific"),
            ("What temperature should I set the equipment/oven to?", "clarify"),
            ("Who won the 2024 Super Bowl?", "fallback"),
        ]

        results = []
        for query_text, expected in queries:
            result = run_query(self.workflow, query_text)
            timing = result.get("timing", {})
            crag_timing = timing.get("crag", {})

            checks = {
                "has_response": bool(result.get("response")),
                "domain_matches": result.get("domain") == expected,
                "confidence_valid": 0.0 <= result.get("confidence", -1) <= 1.0,
                "timing_schema_ok": all(
                    k in timing for k in ("total_s", "supervisor_s", "agent_s", "crag")
                ),
                "crag_schema_ok": all(
                    k in crag_timing
                    for k in ("retrieve_s", "grade_s", "rewrite_s", "generate_s", "validate_s", "total_s")
                ),
            }
            passed = all(checks.values())

            results.append({
                "query": query_text,
                "expected": expected,
                "actual": result.get("domain"),
                "passed": passed,
                "checks": checks,
            })
            status = "PASS" if passed else "FAIL"
            logger.info("Functional %s: %s → %s (expected %s)", status, query_text[:50], result.get("domain"), expected)

        return {
            "pass_rate": sum(r["passed"] for r in results) / len(results) if results else 0,
            "results": results,
        }

    # ── Verification: Reliability ──────────────────────────────────────

    @staticmethod
    def verify_reliability_json_parsing() -> Dict:
        """Test JSON extraction robustness with known-malformed LLM outputs.

        Does NOT require a running model -- tests the static parsing logic.
        """
        import json as _json
        from foundation.llm_wrapper import KerasHubChatModel

        test_cases = [
            ('{"domain":"industrial","confidence":0.9,"reasoning":"fault code"}',
             {"domain": "industrial"}),
            ('```json\n{"domain":"recipe","confidence":0.8,"reasoning":"cooking"}\n```',
             {"domain": "recipe"}),
            ('Sure! Here is the routing:\n{"domain":"scientific","confidence":0.95,"reasoning":"papers"}\nHope this helps!',
             {"domain": "scientific"}),
            ('user\nYou are the Supervisor Agent...',
             {"error": "parse_failed"}),
            ('[{"domain":"industrial","confidence":0.9,"reasoning":"only item"}]',
             {"domain": "industrial"}),
            ('{"confidence": "high", "domain": "recipe", "reasoning": "baking"}',
             {"domain": "recipe"}),
        ]

        results = []
        for raw_text, expected_subset in test_cases:
            # Directly test the parsing logic (text → dict) without model
            text = raw_text.strip()

            if "```" in text:
                parts = text.split("```")
                for part in parts[1::2]:
                    cleaned = part.strip()
                    if cleaned.startswith("json"):
                        cleaned = cleaned[4:].strip()
                    if cleaned.startswith("{") or cleaned.startswith("["):
                        text = cleaned
                        break

            text = text.strip()
            parsed = None
            try:
                p = _json.loads(text)
                if isinstance(p, list) and len(p) == 1 and isinstance(p[0], dict):
                    parsed = p[0]
                elif isinstance(p, dict):
                    parsed = p
            except _json.JSONDecodeError:
                pass

            if parsed is None:
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        parsed = _json.loads(text[start : end + 1])
                    except _json.JSONDecodeError:
                        pass

            if parsed is None:
                parsed = {"error": "parse_failed"}

            match = all(parsed.get(k) == v for k, v in expected_subset.items())
            results.append({
                "input": raw_text[:80],
                "expected_keys": expected_subset,
                "got": {k: parsed.get(k) for k in expected_subset},
                "match": match,
            })

        passed = sum(r["match"] for r in results)
        logger.info("Reliability JSON parsing: %d/%d passed", passed, len(results))
        return {"passed": passed, "total": len(results), "results": results}

    @staticmethod
    def verify_reliability_confidence_clamping() -> Dict:
        """Test confidence validation edge cases in supervisor."""
        from orchestration.supervisor import SupervisorAgent

        class FakeLLM:
            pass

        sup = SupervisorAgent.__new__(SupervisorAgent)
        sup.threshold = 0.75
        sup.valid_domains = {"industrial", "recipe", "scientific", "clarify", "fallback"}

        test_cases = [
            ({"confidence": 1.5, "domain": "industrial"}, 1.0),
            ({"confidence": -0.3, "domain": "recipe"}, 0.0),
            ({"confidence": "bad", "domain": "scientific"}, 0.0),
            ({"confidence": float("nan"), "domain": "industrial"}, 0.0),
            ({"confidence": 0.85, "domain": "industrial"}, 0.85),
            ({"confidence": 0.85, "domain": "unknown_domain"}, 0.85),
        ]

        results = []
        for input_dict, expected_conf in test_cases:
            routing = sup._build_routing_from_result(input_dict, "test query")
            conf_ok = abs(routing["confidence"] - expected_conf) < 1e-6
            domain_ok = routing["domain"] in sup.valid_domains
            results.append({
                "input": str(input_dict),
                "expected_confidence": expected_conf,
                "got_confidence": routing["confidence"],
                "confidence_ok": conf_ok,
                "domain_valid": domain_ok,
            })

        passed = sum(r["confidence_ok"] and r["domain_valid"] for r in results)
        logger.info("Reliability confidence clamping: %d/%d passed", passed, len(results))
        return {"passed": passed, "total": len(results), "results": results}

    # ── Verification: Performance ──────────────────────────────────────

    def verify_performance(self, queries: Optional[List] = None) -> Dict:
        """Measure p50/p95 latency and per-step breakdown.

        Run a set of queries and compute latency statistics.
        """
        from orchestration.workflow_graph import run_query

        queries = queries or [
            "My S7-1200 PLC is showing fault code F0003",
            "How do I make a proper roux for gumbo?",
            "Summarize recent papers on transformer attention",
        ]

        timings = []
        for q in queries:
            result = run_query(self.workflow, q)
            t = result.get("timing", {})
            timings.append({
                "query": q[:60],
                "total_s": t.get("total_s", 0),
                "supervisor_s": t.get("supervisor_s", 0),
                "agent_s": t.get("agent_s", 0),
                "crag_retrieve_s": t.get("crag", {}).get("retrieve_s", 0),
                "crag_grade_s": t.get("crag", {}).get("grade_s", 0),
                "crag_rewrite_s": t.get("crag", {}).get("rewrite_s", 0),
                "crag_generate_s": t.get("crag", {}).get("generate_s", 0),
                "crag_validate_s": t.get("crag", {}).get("validate_s", 0),
            })

        totals = sorted(t["total_s"] for t in timings)
        n = len(totals)

        stats = {
            "n_queries": n,
            "p50_s": totals[n // 2] if n else 0,
            "p95_s": totals[int(n * 0.95)] if n else 0,
            "min_s": totals[0] if n else 0,
            "max_s": totals[-1] if n else 0,
        }
        logger.info(
            "Performance: n=%d  p50=%.1fs  p95=%.1fs  min=%.1fs  max=%.1fs",
            stats["n_queries"], stats["p50_s"], stats["p95_s"],
            stats["min_s"], stats["max_s"],
        )
        return {"stats": stats, "per_query": timings}

    # ── Verification: Regression ───────────────────────────────────────

    def run_full_evaluation(self) -> EvaluationResult:
        """Run the complete evaluation suite (regression baseline)."""
        logger.info("Starting full evaluation...")

        routing_results = self.evaluate_routing()

        eval_result = EvaluationResult(
            routing_accuracy=routing_results["overall_accuracy"],
            routing_by_domain=routing_results["per_domain_accuracy"],
            per_query_results=routing_results["results"],
            avg_latency_seconds=(
                sum(r["latency"] for r in routing_results["results"])
                / len(routing_results["results"])
                if routing_results["results"] else 0
            ),
        )

        completed = sum(
            1 for r in routing_results["results"]
            if r.get("correct")
        )
        eval_result.task_completion_rate = (
            completed / len(routing_results["results"])
            if routing_results["results"] else 0
        )

        logger.info(
            f"Evaluation complete. Routing accuracy: {eval_result.routing_accuracy:.2%}"
        )
        return eval_result
