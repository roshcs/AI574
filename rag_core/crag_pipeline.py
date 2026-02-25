"""
CRAG Pipeline (Corrective RAG)
===============================
Orchestrates the full self-correcting retrieval-augmented generation loop:

    Retrieve → Grade → Decide → Generate → Validate

If retrieved docs are irrelevant, rewrites the query and retries
(up to max_rewrite_attempts). If generation seems ungrounded,
flags for escalation to the supervisor.

This is the SHARED pipeline used by ALL specialized agents.

Usage:
    from rag_core.crag_pipeline import CRAGPipeline
    pipeline = CRAGPipeline(llm=llm, vector_store=vs)
    result = pipeline.run(query="PLC fault F0003", domain="industrial")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from langchain_core.documents import Document

from rag_core.retriever import Retriever
from rag_core.document_grader import DocumentGrader
from rag_core.query_rewriter import QueryRewriter
from rag_core.response_generator import ResponseGenerator
from rag_core.hallucination_checker import HallucinationChecker
from foundation.vector_store import VectorStoreService
from config.settings import CONFIG

logger = logging.getLogger(__name__)


@dataclass
class CRAGResult:
    """Result of a CRAG pipeline execution."""
    response: str = ""
    sources: list = field(default_factory=list)
    domain: str = ""
    grounded: bool = True
    confidence: float = 1.0
    attempts: int = 1
    escalated: bool = False
    escalation_reason: str = ""
    query_history: list = field(default_factory=list)
    grading_summary: dict = field(default_factory=dict)
    timing_breakdown: dict = field(default_factory=dict)
    per_attempt_timing: list = field(default_factory=list)  # [{retrieve_s, grade_s, rewrite_s}, ...]


class CRAGPipeline:
    """
    Self-correcting RAG pipeline implementing the CRAG pattern.

    The pipeline:
    1. RETRIEVE — semantic search in domain collection
    2. GRADE — LLM judges document relevance
    3. DECIDE —
         • Relevant docs found → proceed to generate
         • No relevant docs → rewrite query and retry (max N times)
         • Still nothing → escalate to supervisor
    4. GENERATE — produce grounded response with source attribution
    5. VALIDATE — hallucination check; escalate if ungrounded
    """

    def __init__(
        self,
        llm,
        vector_store: VectorStoreService,
    ):
        self.retriever = Retriever(vector_store)
        self.grader = DocumentGrader(llm)
        self.rewriter = QueryRewriter(llm)
        self.generator = ResponseGenerator(llm)
        self.checker = HallucinationChecker(llm)
        self.max_attempts = CONFIG.rag.max_rewrite_attempts + 1  # 1 initial + N rewrites

    def run(
        self,
        query: str,
        domain: str,
        k: Optional[int] = None,
    ) -> CRAGResult:
        """
        Execute the full CRAG loop.

        Args:
            query: User query (or already rewritten query).
            domain: Target domain for retrieval.
            k: Number of documents to retrieve per attempt.

        Returns:
            CRAGResult with response, sources, and pipeline metadata.
        """
        result = CRAGResult(domain=domain)
        current_query = query
        result.query_history.append(current_query)

        for attempt in range(1, self.max_attempts + 1):
            result.attempts = attempt
            attempt_timing: dict = {}
            logger.info(f"CRAG attempt {attempt}/{self.max_attempts} | "
                        f"Query: {current_query[:80]}")

            # ── Step 1: Retrieve ──────────────────────────────────────
            t0 = time.perf_counter()
            retrieved_docs = self.retriever.retrieve(domain, current_query, k=k)
            retrieve_elapsed = time.perf_counter() - t0
            attempt_timing["retrieve_s"] = retrieve_elapsed
            result.timing_breakdown["retrieve_s"] = result.timing_breakdown.get("retrieve_s", 0) + retrieve_elapsed

            if not retrieved_docs:
                logger.warning("No documents retrieved at all")
                if attempt < self.max_attempts:
                    t0 = time.perf_counter()
                    current_query = self._rewrite_and_retry(
                        current_query, domain, "No documents returned"
                    )
                    rewrite_elapsed = time.perf_counter() - t0
                    attempt_timing["rewrite_s"] = rewrite_elapsed
                    result.timing_breakdown["rewrite_s"] = result.timing_breakdown.get("rewrite_s", 0) + rewrite_elapsed
                    result.per_attempt_timing.append(attempt_timing)
                    result.query_history.append(current_query)
                    continue
                else:
                    result.per_attempt_timing.append(attempt_timing)
                    return self._escalate(
                        result, "No documents found after all attempts"
                    )

            # ── Step 2: Grade ─────────────────────────────────────────
            t0 = time.perf_counter()
            grading = self.grader.grade_documents(current_query, retrieved_docs)
            grade_elapsed = time.perf_counter() - t0
            attempt_timing["grade_s"] = grade_elapsed
            result.timing_breakdown["grade_s"] = result.timing_breakdown.get("grade_s", 0) + grade_elapsed
            result.grading_summary = {
                "relevant": len(grading["relevant"]),
                "irrelevant": len(grading["irrelevant"]),
                "ambiguous": len(grading["ambiguous"]),
            }

            # ── Step 3: Decide ────────────────────────────────────────
            if self.grader.has_sufficient_context(grading):
                context_docs = grading["relevant"] + grading["ambiguous"]
                logger.info(
                    f"Sufficient context found: {len(context_docs)} docs"
                )
                result.per_attempt_timing.append(attempt_timing)
                break
            else:
                logger.info("Insufficient relevant documents")
                if attempt < self.max_attempts:
                    failure_context = (
                        f"Retrieved {len(retrieved_docs)} docs but "
                        f"{len(grading['irrelevant'])} were irrelevant. "
                        f"Grades: {result.grading_summary}"
                    )
                    t0 = time.perf_counter()
                    current_query = self._rewrite_and_retry(
                        current_query, domain, failure_context
                    )
                    rewrite_elapsed = time.perf_counter() - t0
                    attempt_timing["rewrite_s"] = rewrite_elapsed
                    result.timing_breakdown["rewrite_s"] = result.timing_breakdown.get("rewrite_s", 0) + rewrite_elapsed
                    result.per_attempt_timing.append(attempt_timing)
                    result.query_history.append(current_query)
                    continue
                else:
                    result.per_attempt_timing.append(attempt_timing)
                    return self._escalate(
                        result,
                        "Could not find relevant documents after "
                        f"{self.max_attempts} attempts"
                    )
        else:
            return self._escalate(result, "Max attempts exhausted")

        # ── Step 4: Generate ──────────────────────────────────────────
        t0 = time.perf_counter()
        generation = self.generator.generate(query, context_docs, domain)
        result.timing_breakdown["generate_s"] = time.perf_counter() - t0
        result.response = generation["response"]
        result.sources = generation["sources"]

        # ── Step 5: Validate (optional; skip for faster inference) ─────
        if CONFIG.rag.skip_hallucination_check:
            result.grounded = True
            result.confidence = 1.0
            result.timing_breakdown["validate_s"] = 0.0
            logger.info("Hallucination check skipped (skip_hallucination_check=True)")
        else:
            t0 = time.perf_counter()
            validation = self.checker.check(result.response, context_docs)
            result.timing_breakdown["validate_s"] = time.perf_counter() - t0
            result.grounded = validation["grounded"]
            result.confidence = validation["confidence"]
            if validation["should_escalate"]:
                logger.warning(
                    f"Hallucination detected: {validation['issues']}"
                )
                result.escalated = True
                result.escalation_reason = (
                    f"Response may not be fully grounded. "
                    f"Issues: {validation['issues']}"
                )
            else:
                logger.info(
                    f"Response validated (confidence={result.confidence:.2f})"
                )

        result.timing_breakdown["total_s"] = (
            result.timing_breakdown.get("retrieve_s", 0)
            + result.timing_breakdown.get("grade_s", 0)
            + result.timing_breakdown.get("rewrite_s", 0)
            + result.timing_breakdown.get("generate_s", 0)
            + result.timing_breakdown.get("validate_s", 0)
        )
        logger.info(
            "⏱ CRAG timing: retrieve=%.1fs grade=%.1fs rewrite=%.1fs "
            "generate=%.1fs validate=%.1fs total=%.1fs (attempts=%d)",
            result.timing_breakdown.get("retrieve_s", 0),
            result.timing_breakdown.get("grade_s", 0),
            result.timing_breakdown.get("rewrite_s", 0),
            result.timing_breakdown.get("generate_s", 0),
            result.timing_breakdown.get("validate_s", 0),
            result.timing_breakdown["total_s"],
            result.attempts,
        )
        return result

    def _rewrite_and_retry(
        self, query: str, domain: str, failure_context: str
    ) -> str:
        """Rewrite query for the next retrieval attempt."""
        new_query = self.rewriter.rewrite(query, domain, failure_context)
        logger.info(f"Rewritten: '{query[:50]}' → '{new_query[:50]}'")
        return new_query

    @staticmethod
    def _escalate(result: CRAGResult, reason: str) -> CRAGResult:
        """Mark the result as needing supervisor escalation."""
        result.escalated = True
        result.escalation_reason = reason
        result.response = (
            "I wasn't able to find relevant information to answer your question "
            "from my knowledge base. Let me escalate this for further assistance."
        )
        logger.warning(f"Escalating to supervisor: {reason}")
        return result
