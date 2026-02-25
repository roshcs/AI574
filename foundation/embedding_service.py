"""
Embedding Service
=================
Wraps sentence-transformers (gte-large-en-v1.5) behind a clean interface.
Handles batching, device placement, and provides hooks for domain fine-tuning.

Usage:
    from foundation.embedding_service import EmbeddingService
    embedder = EmbeddingService()
    vectors = embedder.embed_documents(["text 1", "text 2"])
    query_vec = embedder.embed_query("search this")
"""

from __future__ import annotations

import logging
from typing import List, Optional

from langchain_core.embeddings import Embeddings

import sys
sys.path.append("..")
from config.settings import EmbeddingConfig, CONFIG

logger = logging.getLogger(__name__)


class EmbeddingService(Embeddings):
    """
    LangChain-compatible embedding service backed by sentence-transformers.
    Implements the Embeddings interface so it plugs directly into ChromaDB,
    retrievers, and any LangChain component expecting an embedder.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or CONFIG.embedding
        self._model = None
        self._load_model()

    def _load_model(self):
        """Load the sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.config.model_name}")
            self._model = SentenceTransformer(
                self.config.model_name,
                device=self.config.device,
                trust_remote_code=self.config.trust_remote_code,
            )
            self._model.max_seq_length = self.config.max_seq_length
            logger.info(
                f"Embedding model loaded. Dimension: {self.config.dimension}, "
                f"Max seq length: {self.config.max_seq_length}"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    # ── LangChain Embeddings Interface ────────────────────────────────────

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents with batching."""
        all_embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            embeddings = self._model.encode(
                batch,
                show_progress_bar=len(texts) > self.config.batch_size,
                normalize_embeddings=True,     # unit vectors for cosine sim
                convert_to_numpy=True,
            )
            all_embeddings.extend(embeddings.tolist())
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding.tolist()

    # ── Domain Fine-Tuning Hook ───────────────────────────────────────────

    def fine_tune(
        self,
        train_pairs: List[tuple],
        epochs: int = 3,
        batch_size: int = 16,
        output_path: str = "./fine_tuned_embeddings",
    ):
        """
        Fine-tune the embedding model on domain-specific (query, positive) pairs
        using contrastive learning with hard negatives.

        Args:
            train_pairs: List of (query, positive_passage) tuples.
            epochs: Training epochs.
            batch_size: Training batch size.
            output_path: Where to save the fine-tuned model.
        """
        from sentence_transformers import InputExample, losses
        from torch.utils.data import DataLoader

        logger.info(
            f"Fine-tuning embeddings on {len(train_pairs)} pairs "
            f"for {epochs} epochs"
        )

        # Build training examples
        examples = [
            InputExample(texts=[q, p]) for q, p in train_pairs
        ]
        loader = DataLoader(examples, shuffle=True, batch_size=batch_size)

        # MultipleNegativesRankingLoss — standard for contrastive fine-tuning
        loss = losses.MultipleNegativesRankingLoss(self._model)

        self._model.fit(
            train_objectives=[(loader, loss)],
            epochs=epochs,
            output_path=output_path,
            show_progress_bar=True,
        )
        logger.info(f"Fine-tuned model saved to {output_path}")

    def load_fine_tuned(self, path: str):
        """Load a previously fine-tuned model."""
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(path, device=self.config.device)
        logger.info(f"Loaded fine-tuned model from {path}")
