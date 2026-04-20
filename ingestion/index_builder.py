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

    # Scientific domain — on-demand backfill from the live ArXiv API
    builder.index_arxiv_papers("transformer attention", max_results=10)

    # Scientific domain — bulk from the Cornell-University/arxiv Kaggle dump
    builder.index_arxiv_dump(
        "./data/scientific/arxiv-metadata-oai-snapshot.json",
        categories=("cs.",),
        year_min=2020,
        batch_size=1000,
    )
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

    def index_recipes(
        self,
        csv_path: str,
        max_rows: Optional[int] = None,
        *,
        dedupe: bool = True,
        drop_junk: bool = True,
        interactions_csv: Optional[str] = None,
        batch_size: int = 1000,
        skip_existing: bool = True,
    ) -> int:
        """
        Index Food.com recipes from CSV.

        Parameters
        ----------
        csv_path : str
            Path to RAW_recipes.csv.
        max_rows : int, optional
            Cap input rows (useful for smoke tests).
        dedupe : bool
            Exact-hash dedupe on normalized (name, ingredients). Default True.
        drop_junk : bool
            Filter rows with empty steps or implausible minutes. Default True.
        interactions_csv : str, optional
            Path to RAW_interactions.csv for popularity enrichment.
        batch_size : int
            How many Documents to embed + upsert per Chroma round-trip.
        skip_existing : bool
            When True (default), recipe IDs already in Chroma are skipped —
            no embedding or upsert is performed for them.  Pass False to
            force a full re-embed (e.g. after changing interaction data).

        Recipes are atomic units: ``ChunkingPipeline`` preserves each recipe
        as a single chunk when it fits the chunk-size budget.
        """
        logger.info(
            f"Indexing recipes from: {csv_path} "
            f"(dedupe={dedupe}, interactions={'yes' if interactions_csv else 'no'}, "
            f"skip_existing={skip_existing})"
        )
        docs = self.loader.load_food_csv(
            csv_path,
            max_rows=max_rows,
            dedupe=dedupe,
            drop_junk=drop_junk,
            interactions_csv=interactions_csv,
        )

        total = 0
        skipped = 0
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            chunks = self.chunker.chunk_documents(batch, domain="recipe")
            before = self.vector_store.get_collection_stats("recipe")["document_count"]
            added = self.vector_store.add_documents(
                "recipe", chunks, skip_existing=skip_existing,
            )
            after = self.vector_store.get_collection_stats("recipe")["document_count"]
            skipped_batch = max(0, len(chunks) - (after - before))
            total += added
            skipped += skipped_batch
            logger.info(
                f"Recipe batch {i // batch_size + 1}: "
                f"added {added}, skipped {skipped_batch} "
                f"({total:,} added / {skipped:,} skipped total)"
            )

        logger.info(
            f"Recipe indexing complete: {total:,} new, {skipped:,} skipped"
        )
        return total

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

    def index_arxiv_dump(
        self,
        dump_path: str,
        max_rows: Optional[int] = None,
        *,
        categories=("cs.",),
        year_min: Optional[int] = 2020,
        year_max: Optional[int] = None,
        dedupe: bool = True,
        batch_size: int = 1000,
        skip_existing: bool = True,
    ) -> int:
        """
        Bulk-index the Cornell-University/arxiv Kaggle dump (NDJSON).

        Parameters
        ----------
        dump_path : str
            Path to the arxiv-metadata-oai-snapshot JSON file.
        max_rows : int, optional
            Cap *kept* rows (after filtering). Useful for smoke tests.
        categories : sequence of str
            Accept papers whose ``categories`` field contains any of these
            prefixes (e.g. ``("cs.",)`` = all computer-science sub-fields).
        year_min, year_max : int, optional
            Year window on ``update_date``. Defaults to 2020..present.
        dedupe : bool
            Drop duplicate arxiv_ids within the stream.
        batch_size : int
            Embed + upsert this many papers per Chroma round-trip.
        skip_existing : bool
            Skip papers whose ``arxiv_<id>`` is already in Chroma, so re-runs
            after interrupted jobs are cheap.

        Returns the count of *new* papers indexed (skipped ones not counted).
        """
        logger.info(
            f"Indexing arxiv dump: {dump_path} "
            f"(categories={list(categories)}, year_min={year_min}, "
            f"year_max={year_max}, skip_existing={skip_existing})"
        )
        docs = self.loader.load_arxiv_dump(
            dump_path,
            categories=categories,
            year_min=year_min,
            year_max=year_max,
            max_rows=max_rows,
            dedupe=dedupe,
        )

        total = 0
        skipped = 0
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            chunks = self.chunker.chunk_documents(batch, domain="scientific")
            before = self.vector_store.get_collection_stats("scientific")["document_count"]
            added = self.vector_store.add_documents(
                "scientific", chunks, skip_existing=skip_existing,
            )
            after = self.vector_store.get_collection_stats("scientific")["document_count"]
            skipped_batch = max(0, len(chunks) - (after - before))
            total += added
            skipped += skipped_batch
            logger.info(
                f"Arxiv batch {i // batch_size + 1}: "
                f"added {added}, skipped {skipped_batch} "
                f"({total:,} added / {skipped:,} skipped total)"
            )

        logger.info(
            f"Arxiv indexing complete: {total:,} new, {skipped:,} skipped"
        )
        return total

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
