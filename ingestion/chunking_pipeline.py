"""
Chunking Pipeline
=================
Splits documents into retrieval-optimized chunks with configurable overlap.
Industrial documents get extra preprocessing (abbreviation expansion,
equipment code normalization).

Usage:
    from ingestion.chunking_pipeline import ChunkingPipeline
    chunker = ChunkingPipeline()
    chunks = chunker.chunk_documents(docs, domain="industrial")
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import ChunkingConfig, CONFIG

logger = logging.getLogger(__name__)


_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ud800-\udfff]")


def _coerce_text(value) -> str:
    """
    Best-effort coercion of arbitrary loader output to a plain Python str.

    Returns '' for empty/whitespace-only input; callers should skip those.
    Also scrubs tokenizer-hostile control chars so downstream HF fast
    tokenizers cannot trip over PDF-extracted NULs / stray surrogates.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    elif isinstance(value, (list, tuple)):
        value = " ".join(_coerce_text(v) for v in value)
    elif not isinstance(value, str):
        try:
            import numpy as np

            if isinstance(value, np.generic):
                value = value.item()
            elif isinstance(value, np.ndarray):
                value = value.ravel().tolist()
                return _coerce_text(value)
        except ImportError:
            pass
        value = str(value)
    if value.__class__ is not str:
        value = str.__str__(value)
    value = _CTRL_RE.sub(" ", value)
    value = value.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    return value.strip()


# ── Industrial Domain Preprocessors ───────────────────────────────────────────

# Common industrial abbreviations → expanded forms
INDUSTRIAL_ABBREVIATIONS: Dict[str, str] = {
    "PLC": "Programmable Logic Controller (PLC)",
    "HMI": "Human-Machine Interface (HMI)",
    "SCADA": "Supervisory Control and Data Acquisition (SCADA)",
    "VFD": "Variable Frequency Drive (VFD)",
    "RTU": "Remote Terminal Unit (RTU)",
    "DCS": "Distributed Control System (DCS)",
    "I/O": "Input/Output (I/O)",
    "OPC": "Open Platform Communications (OPC)",
    "SIL": "Safety Integrity Level (SIL)",
    "MTBF": "Mean Time Between Failures (MTBF)",
    "MTTR": "Mean Time To Repair (MTTR)",
    "PM": "Preventive Maintenance (PM)",
    "CM": "Corrective Maintenance (CM)",
    "NER": "Named Entity Recognition (NER)",
}

# Equipment code patterns (e.g., S7-1200, 6ES7-214, AB-1756)
EQUIPMENT_CODE_PATTERN = re.compile(
    r'\b([A-Z]{1,4}[-/]?\d{1,5}[-/]?\d{0,5}[A-Z]{0,3})\b'
)


def _preprocess_industrial(text: str) -> str:
    """
    Industrial-specific text preprocessing:
    1. Expand abbreviations (first occurrence only to avoid bloat)
    2. Normalize equipment codes for consistent retrieval
    3. Normalize whitespace
    """
    text = _coerce_text(text)
    if not text:
        return ""
    expanded = set()
    for abbrev, full in INDUSTRIAL_ABBREVIATIONS.items():
        if abbrev in text and abbrev not in expanded:
            # Replace first standalone occurrence
            text = re.sub(
                rf'\b{re.escape(abbrev)}\b',
                full,
                text,
                count=1,
            )
            expanded.add(abbrev)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Domain preprocessor registry
_PREPROCESSORS = {
    "industrial": _preprocess_industrial,
}


# Domains whose documents are already atomic units of meaning (one recipe =
# one chunk). For these, only split when the document exceeds the chunk-size
# budget — and then along domain-aware separators.
_ATOMIC_DOMAINS = frozenset({"recipe", "scientific"})

# Separator ordering per atomic domain — used only when the document overflows.
_ATOMIC_SPLIT_SEPARATORS: Dict[str, list] = {
    # Prefer splitting between steps (`\n  N.`) and sections (`\n\n`) before
    # falling back to sentence / whitespace boundaries.
    "recipe": ["\nSteps:\n", "\n\n", "\n  ", "\n", ". ", " "],
    # Arxiv papers are rendered as Title / Authors / ... / Abstract / Comments
    # separated by blank lines; prefer section breaks before sentence splits.
    "scientific": ["\n\n", "\n", ". ", "! ", "? ", " "],
}


# ── Chunking Pipeline ────────────────────────────────────────────────────────

class ChunkingPipeline:
    """
    Document chunking with domain-aware preprocessing.

    Uses RecursiveCharacterTextSplitter under the hood with configurable
    chunk size, overlap, and separators. Industrial documents are preprocessed
    before chunking.
    """

    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or CONFIG.chunking
        # Approximate tokens → chars (1 token ≈ 4 chars)
        self._char_chunk_size = self.config.chunk_size * 4
        self._char_overlap = int(self._char_chunk_size * self.config.chunk_overlap_pct)

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._char_chunk_size,
            chunk_overlap=self._char_overlap,
            separators=self.config.separators,
            length_function=len,
            is_separator_regex=False,
        )

        # Lazily-built splitters for atomic domains with custom separators.
        self._atomic_splitters: Dict[str, RecursiveCharacterTextSplitter] = {}

    def _get_atomic_splitter(self, domain: str) -> RecursiveCharacterTextSplitter:
        """Return (caching) a splitter tuned for an atomic domain's overflow case."""
        if domain not in self._atomic_splitters:
            self._atomic_splitters[domain] = RecursiveCharacterTextSplitter(
                chunk_size=self._char_chunk_size,
                chunk_overlap=self._char_overlap,
                separators=_ATOMIC_SPLIT_SEPARATORS.get(domain, self.config.separators),
                length_function=len,
                is_separator_regex=False,
            )
        return self._atomic_splitters[domain]

    def chunk_documents(
        self,
        documents: List[Document],
        domain: str = "general",
    ) -> List[Document]:
        """
        Chunk a list of documents, applying domain-specific preprocessing.

        Each output chunk inherits the parent document's metadata plus:
        - chunk_index: position within the parent
        - parent_id: ID of the source document
        """
        preprocessor = _PREPROCESSORS.get(domain)
        is_atomic = domain in _ATOMIC_DOMAINS
        atomic_splitter = self._get_atomic_splitter(domain) if is_atomic else None

        all_chunks: List[Document] = []
        atomic_kept = 0
        atomic_split = 0
        dropped_empty = 0

        for doc in documents:
            text = _coerce_text(doc.page_content)
            if not text:
                dropped_empty += 1
                continue
            if preprocessor:
                text = _coerce_text(preprocessor(text))
                if not text:
                    dropped_empty += 1
                    continue

            if is_atomic and len(text) <= self._char_chunk_size:
                text_chunks = [text]
                atomic_kept += 1
            elif is_atomic:
                text_chunks = atomic_splitter.split_text(text)
                atomic_split += 1
            else:
                text_chunks = self._splitter.split_text(text)

            clean_chunks: List[str] = []
            for ct in text_chunks:
                ct = _coerce_text(ct)
                if ct:
                    clean_chunks.append(ct)
            if not clean_chunks:
                dropped_empty += 1
                continue

            parent_id = doc.metadata.get("id", doc.metadata.get("source", "unknown"))

            for i, chunk_text in enumerate(clean_chunks):
                chunk_metadata = {
                    **doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(clean_chunks),
                    "parent_id": parent_id,
                    "id": parent_id if len(clean_chunks) == 1 else f"{parent_id}_chunk_{i}",
                    "domain": domain,
                }
                all_chunks.append(
                    Document(page_content=chunk_text, metadata=chunk_metadata)
                )

        if dropped_empty:
            logger.warning(
                "Chunker dropped %d document(s) with empty/non-text content "
                "(domain=%s)",
                dropped_empty,
                domain,
            )

        if is_atomic:
            logger.info(
                f"Chunked {len(documents)} {domain} documents into "
                f"{len(all_chunks)} chunks (atomic={atomic_kept}, split={atomic_split})"
            )
        else:
            logger.info(
                f"Chunked {len(documents)} documents into {len(all_chunks)} chunks "
                f"(domain={domain})"
            )
        return all_chunks

    def chunk_single(self, text: str, domain: str = "general") -> List[str]:
        """Chunk a single text string. Returns raw strings."""
        text = _coerce_text(text)
        if not text:
            return []
        preprocessor = _PREPROCESSORS.get(domain)
        if preprocessor:
            text = _coerce_text(preprocessor(text))
            if not text:
                return []
        return [_coerce_text(c) for c in self._splitter.split_text(text) if _coerce_text(c)]
