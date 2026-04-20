"""
Vector Store Service
====================
ChromaDB abstraction with multi-domain collection management.
Each domain (industrial, recipe, scientific) gets its own isolated collection.

Usage:
    from foundation.vector_store import VectorStoreService
    vs = VectorStoreService(embedding_service=embedder)
    vs.add_documents("industrial", docs)
    results = vs.search("industrial", "PLC fault code F0003", k=5)
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document

from foundation.embedding_service import EmbeddingService
from config.settings import VectorStoreConfig, CONFIG

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    Multi-domain vector store backed by ChromaDB.

    Key features:
    - One collection per domain for isolated retrieval
    - Consistent interface for add / search / delete
    - LangChain Document in/out for pipeline compatibility
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        config: Optional[VectorStoreConfig] = None,
    ):
        self.config = config or CONFIG.vector_store
        self.embedder = embedding_service
        self._client = None
        self._collections: Dict[str, chromadb.Collection] = {}
        self._initialize()

    def _initialize(self):
        """Initialize ChromaDB client and create collections."""
        persist_dir = os.getenv("CHROMA_PERSIST_DIR") or self.config.persist_directory
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # Create or get each domain collection
        for domain, collection_name in self.config.collections.items():
            self._collections[domain] = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": self.config.similarity_metric},
            )
            logger.info(
                f"Collection '{collection_name}' ready "
                f"(count: {self._collections[domain].count()})"
            )

    # ── Core Operations ───────────────────────────────────────────────────

    def add_documents(
        self,
        domain: str,
        documents: List[Document],
        batch_size: int = 100,
        skip_existing: bool = False,
    ) -> int:
        """
        Add LangChain Documents to a domain-specific collection.

        Parameters
        ----------
        skip_existing : bool
            When True, IDs already present in the collection are silently
            skipped — no embedding or upsert is performed for them.  This
            makes re-runs cheap: only genuinely new documents pay for
            embedding compute.

        Returns the number of documents actually added.
        """
        self._validate_domain(domain)
        collection = self._collections[domain]

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [
            doc.metadata.get("id", f"{domain}_{i}")
            for i, doc in enumerate(documents)
        ]

        added = 0
        skipped = 0
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            keep = list(range(len(batch_ids)))

            if skip_existing:
                seen: set = set()
                deduped = []
                for j in keep:
                    if batch_ids[j] not in seen:
                        seen.add(batch_ids[j])
                        deduped.append(j)
                keep = deduped

                present = set(
                    collection.get(
                        ids=[batch_ids[j] for j in keep],
                        include=[],
                    )["ids"]
                )
                keep = [j for j in keep if batch_ids[j] not in present]

            if not keep:
                skipped += len(batch_ids)
                continue

            final_texts = [batch_texts[j] for j in keep]
            final_meta = [batch_meta[j] for j in keep]
            final_ids = [batch_ids[j] for j in keep]
            final_embeddings = self.embedder.embed_documents(final_texts)

            collection.upsert(
                ids=final_ids,
                embeddings=final_embeddings,
                documents=final_texts,
                metadatas=final_meta,
            )
            added += len(final_ids)
            skipped += len(batch_ids) - len(final_ids)

        if skipped:
            logger.info(
                f"Added {added}, skipped {skipped} existing "
                f"documents in '{domain}' collection"
            )
        else:
            logger.info(f"Added {added} documents to '{domain}' collection")
        return added

    def _embed_query_cached(self, query: str) -> Tuple[float, ...]:
        """Cache query embeddings to avoid re-encoding on CRAG retries."""
        return self.__embed_cache(query)

    @lru_cache(maxsize=128)
    def __embed_cache(self, query: str) -> Tuple[float, ...]:
        return tuple(self.embedder.embed_query(query))

    def search(
        self,
        domain: str,
        query: str,
        k: Optional[int] = None,
        where: Optional[dict] = None,
    ) -> List[Document]:
        """
        Semantic search within a domain collection.

        Returns LangChain Documents with metadata including similarity score.
        """
        self._validate_domain(domain)
        k = k or self.config.search_top_k
        collection = self._collections[domain]

        query_embedding = list(self._embed_query_cached(query))

        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed for domain '{domain}': {e}")
            return []

        if not results or not results.get("documents") or not results["documents"][0]:
            return []

        documents = []
        for doc_text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            metadata["similarity_score"] = 1 - distance  # cosine distance → sim
            documents.append(
                Document(page_content=doc_text, metadata=metadata)
            )

        return documents

    def get_collection_stats(self, domain: str) -> dict:
        """Get stats for a domain collection."""
        self._validate_domain(domain)
        collection = self._collections[domain]
        return {
            "domain": domain,
            "collection_name": collection.name,
            "document_count": collection.count(),
        }

    def get_all_stats(self) -> List[dict]:
        """Get stats for all domain collections."""
        return [self.get_collection_stats(d) for d in self._collections]

    def clear_collection(self, domain: str):
        """Delete all documents in a domain collection."""
        self._validate_domain(domain)
        collection_name = self.config.collections[domain]
        self._client.delete_collection(collection_name)
        self._collections[domain] = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": self.config.similarity_metric},
        )
        logger.info(f"Cleared collection for domain '{domain}'")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _validate_domain(self, domain: str):
        if domain not in self._collections:
            raise ValueError(
                f"Unknown domain '{domain}'. "
                f"Available: {list(self._collections.keys())}"
            )
