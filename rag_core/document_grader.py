"""
Document Grader
===============
LLM-as-judge that evaluates whether retrieved documents are relevant
to the user's query. Core component of the CRAG self-correction loop.

Usage:
    from rag_core.document_grader import DocumentGrader
    grader = DocumentGrader(llm=llm)
    results = grader.grade_documents(query, retrieved_docs)
    relevant_docs = results["relevant"]
"""

from __future__ import annotations

import logging
from typing import Dict, List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from config.prompts import GRADER_PROMPT, GRADER_BATCH_PROMPT
from config.settings import CONFIG

logger = logging.getLogger(__name__)


BATCH_GRADING_CHUNK_SIZE = 3


class DocumentGrader:
    """
    Grades retrieved documents for relevance using LLM-as-judge.
    Categorizes each document as relevant, irrelevant, or ambiguous.
    """

    def __init__(self, llm):
        """
        Args:
            llm: A LangChain-compatible chat model (e.g., KerasHubChatModel).
        """
        self.llm = llm
        self.threshold = CONFIG.rag.relevance_threshold
        self._batch_fallback_count = 0
        self._batch_total_count = 0

    def grade_single(self, query: str, document: Document) -> dict:
        """
        Grade a single document's relevance to the query.

        Returns:
            {"relevance": "relevant"|"irrelevant"|"ambiguous",
             "score": float, "reasoning": str}
        """
        prompt = GRADER_PROMPT.format(
            query=query,
            document=document.page_content[:2000],  # truncate for context window
        )

        try:
            result = self.llm.invoke_and_parse_json(
                [HumanMessage(content=prompt)]
            )
            if "error" in result:
                # Parse failed — fall back to score-based grading
                logger.warning("Grader JSON parse failed, using fallback")
                return {
                    "relevance": "ambiguous",
                    "score": 0.5,
                    "reasoning": "Parse error — defaulting to ambiguous",
                }
            return result
        except Exception as e:
            logger.error(f"Grading failed: {e}")
            return {
                "relevance": "ambiguous",
                "score": 0.5,
                "reasoning": f"Error: {str(e)}",
            }

    def grade_documents_batch(
        self,
        query: str,
        documents: List[Document],
    ) -> Dict[str, List[Document]]:
        """
        Grade all documents in a single LLM call (faster than N separate calls).
        Falls back to per-document grading if batch parsing fails.
        """
        if not documents:
            return {"relevant": [], "irrelevant": [], "ambiguous": [], "grades": []}

        documents_block = "\n\n---\n\n".join(
            f"[Document {i}]\n{doc.page_content[:2000]}"
            for i, doc in enumerate(documents)
        )
        prompt = GRADER_BATCH_PROMPT.format(
            query=query,
            documents_block=documents_block,
        )

        try:
            result = self.llm.invoke_and_parse_json([HumanMessage(content=prompt)])
            if "error" in result or "grades" not in result:
                raise ValueError("Batch grader returned no grades")
            raw_grades = result["grades"]
            if not isinstance(raw_grades, list):
                raise ValueError("Grades field is not a list")
            if len(raw_grades) < len(documents):
                logger.warning(
                    f"Partial batch result: expected {len(documents)} grades, "
                    f"got {len(raw_grades)}; padding remainder as ambiguous"
                )
        except Exception as e:
            self._batch_fallback_count += 1
            logger.warning(f"Batch grading failed ({e}), falling back to per-document grading")
            return self._grade_documents_sequential(query, documents)

        buckets = {"relevant": [], "irrelevant": [], "ambiguous": []}
        grades = []
        for i, doc in enumerate(documents):
            if i < len(raw_grades) and isinstance(raw_grades[i], dict):
                g = raw_grades[i]
            else:
                g = {"relevance": "ambiguous", "score": 0.5, "reasoning": "Missing from batch output"}
            relevance = g.get("relevance", "ambiguous")
            if relevance not in CONFIG.rag.grading_labels:
                relevance = "ambiguous"
            buckets[relevance].append(doc)
            grades.append({"doc_index": i, "grade": g})
            logger.debug(f"Doc {i}: {relevance} (score={g.get('score', '?')})")

        logger.info(
            f"Grading results — relevant: {len(buckets['relevant'])}, "
            f"irrelevant: {len(buckets['irrelevant'])}, "
            f"ambiguous: {len(buckets['ambiguous'])}"
        )
        return {**buckets, "grades": grades}

    def _grade_documents_sequential(
        self,
        query: str,
        documents: List[Document],
    ) -> Dict[str, List[Document]]:
        """Per-document grading (slower, used as fallback)."""
        buckets = {"relevant": [], "irrelevant": [], "ambiguous": []}
        grades = []
        for i, doc in enumerate(documents):
            grade = self.grade_single(query, doc)
            relevance = grade.get("relevance", "ambiguous")
            if relevance not in CONFIG.rag.grading_labels:
                relevance = "ambiguous"
            buckets[relevance].append(doc)
            grades.append({"doc_index": i, "grade": grade})
        logger.info(
            f"Grading results — relevant: {len(buckets['relevant'])}, "
            f"irrelevant: {len(buckets['irrelevant'])}, ambiguous: {len(buckets['ambiguous'])}"
        )
        return {**buckets, "grades": grades}

    def grade_documents(
        self,
        query: str,
        documents: List[Document],
    ) -> Dict[str, List[Document]]:
        """
        Grade all retrieved documents and bucket them.

        Splits large batches into chunks of BATCH_GRADING_CHUNK_SIZE to reduce
        the likelihood of LLM output truncation, then merges the per-chunk
        results. Falls back to sequential grading on a per-chunk basis.
        """
        if not documents:
            return {"relevant": [], "irrelevant": [], "ambiguous": [], "grades": []}

        chunk_size = BATCH_GRADING_CHUNK_SIZE
        if len(documents) <= chunk_size:
            return self.grade_documents_batch(query, documents)

        merged: Dict[str, list] = {"relevant": [], "irrelevant": [], "ambiguous": [], "grades": []}
        for start in range(0, len(documents), chunk_size):
            chunk = documents[start : start + chunk_size]
            self._batch_total_count += 1
            chunk_result = self.grade_documents_batch(query, chunk)

            for label in ("relevant", "irrelevant", "ambiguous"):
                merged[label].extend(chunk_result[label])
            for g in chunk_result.get("grades", []):
                merged["grades"].append({
                    "doc_index": g["doc_index"] + start,
                    "grade": g["grade"],
                })

        logger.info(
            f"Chunked grading complete: {len(merged['grades'])} docs across "
            f"{(len(documents) + chunk_size - 1) // chunk_size} chunks "
            f"(batch fallbacks so far: {self._batch_fallback_count}/{self._batch_total_count})"
        )
        return merged

    def has_sufficient_context(self, grading_result: dict) -> bool:
        """Check if enough relevant documents were found to generate a response."""
        return len(grading_result["relevant"]) >= 1
