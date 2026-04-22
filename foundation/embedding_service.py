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
import re
from typing import Any, List, Optional

from langchain_core.embeddings import Embeddings

import sys
sys.path.append("..")
from config.settings import EmbeddingConfig, CONFIG, DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

# C0 control chars (except \t, \n, \r) + DEL + stray surrogates. HF fast
# tokenizers reject some of these with the cryptic "TextEncodeInput must be
# Union[...]" error. PyPDF extraction is the usual source.
_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff]"
)


def _strip_controls(s: str) -> str:
    """Replace tokenizer-hostile control chars with a space."""
    return _CONTROL_CHAR_RE.sub(" ", s)


def _normalize_one(value: Any, *, _depth: int = 0) -> str:
    """
    Coerce arbitrary loader output to a plain non-empty Python str for tokenizers.

    Handles bytes, nested lists/tuples, numpy scalars/arrays, and other types
    that break HuggingFace fast tokenizers (TextEncodeInput TypeError).
    """
    if _depth > 12:
        return " "
    if value is None:
        return " "
    if isinstance(value, bytes):
        s = value.decode("utf-8", errors="replace").strip()
        return s if s else " "
    if isinstance(value, (list, tuple)):
        parts = [_normalize_one(x, _depth=_depth + 1) for x in value]
        joined = " ".join(p for p in parts if p and p != " ")
        return joined if joined.strip() else " "
    try:
        import numpy as np

        if isinstance(value, np.generic):
            value = value.item()
            return _normalize_one(value, _depth=_depth + 1)
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return " "
            if value.ndim == 0:
                return _normalize_one(value.item(), _depth=_depth + 1)
            return _normalize_one(value.ravel().tolist(), _depth=_depth + 1)
    except ImportError:
        pass
    if not isinstance(value, str):
        value = str(value)
    # Cast to plain `str` (not numpy.str_, not subclass) and scrub control chars.
    value = str.__str__(value) if value.__class__ is not str else value
    value = _strip_controls(value)
    # Re-encode through UTF-8 with replacement to drop invalid surrogates.
    value = value.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    value = value.strip()
    return value if value else " "


def _sanitize_embedding_texts(texts: List[str]) -> List[str]:
    """List-wise wrapper: plain str instances for sentence-transformers."""
    return [str.__str__(_normalize_one(t)) for t in texts]


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

    @staticmethod
    def _validate_model_name(name: str) -> str:
        """Return a valid HuggingFace model id, falling back to the default."""
        if not isinstance(name, str) or not name.strip():
            logger.warning(
                "model_name is empty or not a string; "
                f"falling back to default '{DEFAULT_EMBEDDING_MODEL}'"
            )
            return DEFAULT_EMBEDDING_MODEL

        stripped = name.strip()
        config_indicators = ('"architectures"', '"model_type"')
        looks_like_config = all(ind in stripped for ind in config_indicators)
        starts_with_config_class = any(
            stripped.startswith(prefix)
            for prefix in ("BertConfig", "NewConfig", "RobertaConfig",
                           "AlbertConfig", "DistilBertConfig", "XLNetConfig")
        )

        if looks_like_config or starts_with_config_class or "\n" in stripped:
            logger.warning(
                "model_name appears to be a serialized config object, not a "
                f"model id; falling back to default '{DEFAULT_EMBEDDING_MODEL}'"
            )
            return DEFAULT_EMBEDDING_MODEL

        return stripped

    def _load_model(self):
        """Load the sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer

            model_name = self._validate_model_name(self.config.model_name)
            logger.info(f"Loading embedding model: {model_name}")
            self._model = SentenceTransformer(
                model_name,
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
        texts = _sanitize_embedding_texts(texts)
        all_embeddings = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = [str(x) for x in texts[i : i + self.config.batch_size]]
            try:
                embeddings = self._model.encode(
                    list(batch),
                    show_progress_bar=len(texts) > self.config.batch_size,
                    normalize_embeddings=True,     # unit vectors for cosine sim
                    convert_to_numpy=True,
                )
            except TypeError:
                logger.warning(
                    "Batch embedding failed; encoding %d texts one-by-one",
                    len(batch),
                )
                import numpy as np

                rows = []
                for idx, s in enumerate(batch):
                    try:
                        v = self._model.encode(
                            s,
                            show_progress_bar=False,
                            normalize_embeddings=True,
                            convert_to_numpy=True,
                        )
                    except TypeError as e:
                        preview = s[:80] if isinstance(s, str) else repr(s)[:80]
                        logger.warning(
                            "Tokenizer rejected chunk %d (type=%s, len=%d, preview=%r): %s "
                            "- substituting zero vector",
                            idx,
                            type(s).__name__,
                            len(s) if hasattr(s, "__len__") else -1,
                            preview,
                            e,
                        )
                        v = np.zeros(self.config.dimension, dtype=np.float32)
                    rows.append(np.atleast_2d(v))
                embeddings = np.vstack(rows)
            all_embeddings.extend(embeddings.tolist())
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        text = _sanitize_embedding_texts([text])[0]
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
