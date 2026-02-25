"""
Tests for CRAG pipeline edge cases, grading robustness, and vector store fallbacks.
Uses lightweight mocks — no GPU or real LLM required.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from langchain_core.documents import Document

from rag_core.crag_pipeline import CRAGPipeline, CRAGResult
from rag_core.document_grader import DocumentGrader, BATCH_GRADING_CHUNK_SIZE
from foundation.vector_store import VectorStoreService


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_docs(n: int = 3) -> list[Document]:
    return [
        Document(page_content=f"Content for doc {i}", metadata={"source": f"src_{i}", "id": f"doc_{i}"})
        for i in range(n)
    ]


def _mock_llm(json_response: dict | None = None, text_response: str = "OK"):
    """Return a mock LLM that responds to invoke / invoke_and_parse_json."""
    llm = MagicMock()
    llm.invoke.return_value = SimpleNamespace(content=text_response)
    llm.invoke_and_parse_json.return_value = json_response or {}
    return llm


def _mock_vector_store():
    """Return a mock VectorStoreService."""
    vs = MagicMock(spec=VectorStoreService)
    vs.search.return_value = _make_docs(3)
    return vs


# ── CRAG Pipeline Tests ─────────────────────────────────────────────────────

class TestCRAGPipelineEdgeCases(unittest.TestCase):

    def test_empty_query_escalates(self):
        llm = _mock_llm()
        vs = _mock_vector_store()
        pipeline = CRAGPipeline(llm=llm, vector_store=vs)
        result = pipeline.run(query="", domain="industrial")
        self.assertTrue(result.escalated)
        self.assertIn("Empty", result.escalation_reason)

    def test_whitespace_query_escalates(self):
        llm = _mock_llm()
        vs = _mock_vector_store()
        pipeline = CRAGPipeline(llm=llm, vector_store=vs)
        result = pipeline.run(query="   \n  ", domain="industrial")
        self.assertTrue(result.escalated)

    def test_no_docs_all_attempts_escalates(self):
        llm = _mock_llm(text_response="rewritten query")
        vs = _mock_vector_store()
        vs.search.return_value = []
        pipeline = CRAGPipeline(llm=llm, vector_store=vs)
        result = pipeline.run(query="some query", domain="industrial")
        self.assertTrue(result.escalated)
        self.assertIn("No documents found", result.escalation_reason)

    def test_successful_pipeline_returns_response(self):
        grading_json = {
            "grades": [
                {"relevance": "relevant", "score": 0.9, "reasoning": "good"},
                {"relevance": "relevant", "score": 0.8, "reasoning": "ok"},
                {"relevance": "irrelevant", "score": 0.2, "reasoning": "off-topic"},
            ]
        }
        hallucination_json = {"grounded": True, "confidence": 0.95, "issues": []}
        llm = _mock_llm(text_response="Generated answer.")
        call_count = 0

        def side_effect(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return grading_json
            return hallucination_json

        llm.invoke_and_parse_json.side_effect = side_effect

        vs = _mock_vector_store()
        pipeline = CRAGPipeline(llm=llm, vector_store=vs)
        result = pipeline.run(query="How to fix PLC error?", domain="industrial")
        self.assertFalse(result.escalated)
        self.assertEqual(result.response, "Generated answer.")
        self.assertTrue(result.grounded)

    def test_rewrite_failure_keeps_original_query(self):
        llm = _mock_llm()
        llm.invoke.side_effect = RuntimeError("LLM down")
        vs = _mock_vector_store()
        pipeline = CRAGPipeline(llm=llm, vector_store=vs)
        original = "PLC fault F0003"
        rewritten = pipeline._rewrite_and_retry(original, "industrial", "no docs")
        self.assertEqual(rewritten, original)


# ── Document Grader Tests ────────────────────────────────────────────────────

class TestDocumentGrader(unittest.TestCase):

    def test_empty_documents_returns_empty_buckets(self):
        llm = _mock_llm()
        grader = DocumentGrader(llm)
        result = grader.grade_documents("query", [])
        self.assertEqual(result["relevant"], [])
        self.assertEqual(result["grades"], [])

    def test_partial_batch_pads_with_ambiguous(self):
        docs = _make_docs(4)
        partial_grades = {
            "grades": [
                {"relevance": "relevant", "score": 0.9, "reasoning": "ok"},
                {"relevance": "irrelevant", "score": 0.1, "reasoning": "bad"},
            ]
        }
        llm = _mock_llm(json_response=partial_grades)
        grader = DocumentGrader(llm)
        result = grader.grade_documents_batch("query", docs)
        self.assertEqual(len(result["grades"]), 4)
        self.assertEqual(result["grades"][2]["grade"]["relevance"], "ambiguous")
        self.assertEqual(result["grades"][3]["grade"]["relevance"], "ambiguous")

    def test_batch_parse_failure_falls_back_to_sequential(self):
        docs = _make_docs(2)
        llm = _mock_llm()
        llm.invoke_and_parse_json.side_effect = [
            RuntimeError("parse error"),
            {"relevance": "relevant", "score": 0.9, "reasoning": "ok"},
            {"relevance": "irrelevant", "score": 0.2, "reasoning": "bad"},
        ]
        grader = DocumentGrader(llm)
        result = grader.grade_documents_batch("query", docs)
        self.assertEqual(len(result["grades"]), 2)
        self.assertEqual(grader._batch_fallback_count, 1)

    def test_chunked_grading_splits_large_batches(self):
        n = BATCH_GRADING_CHUNK_SIZE * 2 + 1
        docs = _make_docs(n)

        def make_grades(messages):
            num = min(BATCH_GRADING_CHUNK_SIZE, n)
            return {
                "grades": [
                    {"relevance": "relevant", "score": 0.8, "reasoning": "ok"}
                    for _ in range(num)
                ]
            }

        llm = _mock_llm()
        llm.invoke_and_parse_json.side_effect = make_grades
        grader = DocumentGrader(llm)
        result = grader.grade_documents("query", docs)
        self.assertEqual(len(result["grades"]), n)

    def test_invalid_relevance_label_defaults_to_ambiguous(self):
        docs = _make_docs(1)
        llm = _mock_llm(json_response={
            "grades": [{"relevance": "UNKNOWN_LABEL", "score": 0.5, "reasoning": "weird"}]
        })
        grader = DocumentGrader(llm)
        result = grader.grade_documents_batch("query", docs)
        self.assertEqual(len(result["ambiguous"]), 1)


# ── Vector Store Tests ───────────────────────────────────────────────────────

class TestVectorStoreSearchFailure(unittest.TestCase):

    @patch("foundation.vector_store.chromadb")
    @patch("foundation.vector_store.EmbeddingService")
    def test_search_returns_empty_on_chromadb_error(self, mock_embed_cls, mock_chromadb):
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 1024

        mock_collection = MagicMock()
        mock_collection.query.side_effect = RuntimeError("disk full")
        mock_collection.count.return_value = 0

        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.Client.return_value = mock_client

        vs = VectorStoreService(embedding_service=mock_embedder)
        results = vs.search("industrial", "test query")
        self.assertEqual(results, [])

    @patch("foundation.vector_store.chromadb")
    @patch("foundation.vector_store.EmbeddingService")
    def test_search_returns_empty_on_no_results(self, mock_embed_cls, mock_chromadb):
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1] * 1024

        mock_collection = MagicMock()
        mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        mock_collection.count.return_value = 0

        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_chromadb.Client.return_value = mock_client

        vs = VectorStoreService(embedding_service=mock_embedder)
        results = vs.search("industrial", "test query")
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
