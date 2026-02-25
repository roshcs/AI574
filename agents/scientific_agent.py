"""
Scientific Paper Summarizer Agent
==================================
Specialized for academic literature synthesis via ArXiv.
Supports on-demand paper retrieval and citation-aware summarization.
"""

from __future__ import annotations

import logging
from typing import Optional

from agents.base_agent import BaseAgent
from rag_core.crag_pipeline import CRAGPipeline, CRAGResult
from ingestion.index_builder import IndexBuilder

logger = logging.getLogger(__name__)


class ScientificAgent(BaseAgent):
    """
    Scientific Paper Summarization Specialist.

    Enhancements over base agent:
    - On-demand ArXiv fetching when local index is insufficient
    - Adds citation formatting to responses
    - Structures summaries as Objective → Method → Findings → Limitations
    """

    def __init__(
        self,
        crag_pipeline: CRAGPipeline,
        index_builder: Optional[IndexBuilder] = None,
    ):
        super().__init__(crag_pipeline)
        self.index_builder = index_builder

    @property
    def domain(self) -> str:
        return "scientific"

    @property
    def description(self) -> str:
        return (
            "Scientific paper summarization specialist. Handles ArXiv research "
            "queries, literature synthesis, and academic concept explanation."
        )

    def handle(self, query: str, **kwargs) -> CRAGResult:
        """
        Extended handle: if local retrieval fails AND we have an index builder,
        fetch fresh papers from ArXiv and retry.
        """
        # First, try with existing indexed papers
        result = super().handle(query, **kwargs)

        # If escalated due to no relevant docs, try fetching from ArXiv
        if (
            result.escalated
            and self.index_builder
            and "No documents" in result.escalation_reason
        ):
            logger.info("Local retrieval failed — fetching from ArXiv on-demand")
            try:
                count = self.index_builder.index_arxiv_papers(
                    query, max_results=10
                )
                if count > 0:
                    logger.info(f"Indexed {count} new ArXiv papers, retrying...")
                    result = super().handle(query, **kwargs)
            except Exception as e:
                logger.error(f"ArXiv on-demand fetch failed: {e}")

        return result

    def preprocess_query(self, query: str) -> str:
        """
        Scientific-specific query preprocessing:
        - Strip conversational fluff for better retrieval
        - Add "research paper" context if not present
        """
        # Remove conversational prefixes
        strip_prefixes = [
            "can you find", "please summarize", "i want to know about",
            "tell me about", "search for", "find papers on",
        ]
        processed = query
        query_lower = query.lower()
        for prefix in strip_prefixes:
            if query_lower.startswith(prefix):
                processed = query[len(prefix):].strip()
                break

        return processed

    def postprocess_response(self, result: CRAGResult) -> CRAGResult:
        """
        Add citation block at the end of response if sources have ArXiv IDs.
        """
        arxiv_sources = [
            s for s in result.sources
            if "arxiv" in s.get("source", "").lower()
        ]

        if arxiv_sources:
            citations = []
            for i, src in enumerate(arxiv_sources, 1):
                parent_id = src.get("parent_id", "")
                citations.append(f"  [{i}] {parent_id}")

            if citations:
                result.response += "\n\n📚 **References:**\n" + "\n".join(citations)

        return result
