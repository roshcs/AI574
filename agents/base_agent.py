"""
Base Agent
==========
Abstract base class for all specialized agents.
Each agent = domain config + optional preprocessing + shared CRAG pipeline.

Subclasses only need to override:
- domain (str)
- preprocess_query() — optional domain-specific query prep
- postprocess_response() — optional domain-specific formatting
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from rag_core.crag_pipeline import CRAGPipeline, CRAGResult

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for domain-specialized agents."""

    def __init__(self, crag_pipeline: CRAGPipeline):
        self.crag = crag_pipeline

    @property
    @abstractmethod
    def domain(self) -> str:
        """Domain identifier (matches vector store collection key)."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of this agent's capabilities."""
        ...

    def handle(self, query: str, **kwargs) -> CRAGResult:
        """
        Full agent execution: preprocess → CRAG → postprocess.

        This is the main entry point called by the supervisor.
        """
        logger.info(f"[{self.domain}] Handling query: {query[:80]}...")

        # Domain-specific query preprocessing
        processed_query = self.preprocess_query(query)

        # Run shared CRAG pipeline
        result = self.crag.run(
            query=processed_query,
            domain=self.domain,
            **kwargs,
        )

        # Domain-specific postprocessing
        result = self.postprocess_response(result)

        return result

    def preprocess_query(self, query: str) -> str:
        """
        Optional domain-specific query preprocessing.
        Override in subclass to add terminology expansion, etc.
        Default: pass-through.
        """
        return query

    def postprocess_response(self, result: CRAGResult) -> CRAGResult:
        """
        Optional domain-specific response postprocessing.
        Override in subclass to format output, add warnings, etc.
        Default: pass-through.
        """
        return result
