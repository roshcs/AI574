"""
Index Builder
=============
Orchestrates the full ingestion pipeline: load → chunk → embed → store.
One-time setup per domain. Provides convenience methods for each domain type.

Usage:
    from ingestion.index_builder import IndexBuilder
    builder = IndexBuilder(vector_store=vs)

    # Industrial domain
    builder.index_industrial_docs("./data/manuals/")

    # Recipe domain
    builder.index_recipes("./data/RAW_recipes.csv", max_rows=50000)

    # Scientific domain (on-demand, not pre-indexed)
    builder.index_arxiv_papers("transformer attention", max_results=10)
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.documents import Document

from ingestion.document_loader import DocumentLoader
from ingestion.chunking_pipeline import ChunkingPipeline
from foundation.vector_store import VectorStoreService

logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    End-to-end indexing orchestrator.
    Connects document loading, chunking, and vector storage.
    """

    def __init__(
        self,
        vector_store: VectorStoreService,
        chunker: Optional[ChunkingPipeline] = None,
    ):
        self.vector_store = vector_store
        self.chunker = chunker or ChunkingPipeline()
        self.loader = DocumentLoader()

    # ── Industrial Domain ─────────────────────────────────────────────────

    def index_industrial_docs(self, directory: str) -> int:
        """
        Index all PDFs and text files from an industrial manuals directory.
        Applies industrial-specific preprocessing during chunking.
        """
        logger.info(f"Indexing industrial documents from: {directory}")
        docs = self.loader.load_directory(directory)
        chunks = self.chunker.chunk_documents(docs, domain="industrial")
        count = self.vector_store.add_documents("industrial", chunks)
        logger.info(f"Industrial indexing complete: {count} chunks stored")
        return count

    def index_industrial_texts(self, texts: List[str], source: str = "manual") -> int:
        """Index raw text strings as industrial documents."""
        docs = self.loader.from_texts(texts, domain="industrial", source=source)
        chunks = self.chunker.chunk_documents(docs, domain="industrial")
        return self.vector_store.add_documents("industrial", chunks)

    # ── Recipe Domain ─────────────────────────────────────────────────────

    def index_recipes(self, csv_path: str, max_rows: Optional[int] = None) -> int:
        """
        Index Food.com recipes from CSV.
        Recipes are already semi-structured so chunking is lighter.
        """
        logger.info(f"Indexing recipes from: {csv_path}")
        docs = self.loader.load_food_csv(csv_path, max_rows=max_rows)
        # Recipes are typically short enough to be single chunks,
        # but we chunk anyway for consistency
        chunks = self.chunker.chunk_documents(docs, domain="recipe")
        count = self.vector_store.add_documents("recipe", chunks)
        logger.info(f"Recipe indexing complete: {count} chunks stored")
        return count

    # ── Scientific Domain ─────────────────────────────────────────────────

    def index_arxiv_papers(
        self,
        query: str,
        max_results: int = 10,
    ) -> int:
        """
        Fetch and index ArXiv papers for a given query.
        This is typically called on-demand rather than as a one-time setup.
        """
        logger.info(f"Fetching ArXiv papers for: '{query}'")
        docs = self.loader.load_arxiv(query, max_results=max_results)
        chunks = self.chunker.chunk_documents(docs, domain="scientific")
        count = self.vector_store.add_documents("scientific", chunks)
        logger.info(f"Scientific indexing complete: {count} chunks stored")
        return count

    # ── Generic ───────────────────────────────────────────────────────────

    def index_documents(
        self,
        documents: List[Document],
        domain: str,
    ) -> int:
        """Index pre-loaded documents into a specific domain."""
        chunks = self.chunker.chunk_documents(documents, domain=domain)
        return self.vector_store.add_documents(domain, chunks)

    def get_status(self) -> dict:
        """Return indexing status across all domains."""
        return {
            "collections": self.vector_store.get_all_stats(),
        }
