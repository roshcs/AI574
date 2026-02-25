"""
Retriever
=========
Thin wrapper around VectorStoreService.search() that provides a consistent
interface for the CRAG pipeline. Adds logging and query preprocessing.

Usage:
    from rag_core.retriever import Retriever
    retriever = Retriever(vector_store=vs)
    docs = retriever.retrieve("industrial", "PLC fault F0003", k=5)
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.documents import Document

from foundation.vector_store import VectorStoreService
from config.settings import CONFIG

logger = logging.getLogger(__name__)


class Retriever:
    """
    Domain-aware document retriever.
    Routes queries to the correct vector store collection.
    """

    def __init__(self, vector_store: VectorStoreService):
        self.vector_store = vector_store

    def retrieve(
        self,
        domain: str,
        query: str,
        k: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> List[Document]:
        """
        Retrieve top-k documents for a query from the specified domain.

        Args:
            domain: Target domain ("industrial", "recipe", "scientific").
            query: The user query or rewritten query.
            k: Number of results (defaults to config).
            filters: Optional ChromaDB where-clause for metadata filtering.

        Returns:
            List of Documents with similarity_score in metadata.
        """
        if not query or not query.strip():
            logger.warning("Empty query passed to retriever, returning no results")
            return []

        k = k or CONFIG.vector_store.search_top_k

        logger.info(f"Retrieving top-{k} from '{domain}' for: {query[:80]}...")
        docs = self.vector_store.search(domain, query, k=k, where=filters)
        logger.info(
            f"Retrieved {len(docs)} documents "
            f"(scores: {[f'{d.metadata.get('similarity_score', 0):.3f}' for d in docs]})"
        )
        return docs

    def retrieve_with_scores(
        self,
        domain: str,
        query: str,
        k: Optional[int] = None,
    ) -> List[tuple]:
        """Retrieve documents as (Document, score) tuples."""
        docs = self.retrieve(domain, query, k=k)
        return [
            (doc, doc.metadata.get("similarity_score", 0.0))
            for doc in docs
        ]
