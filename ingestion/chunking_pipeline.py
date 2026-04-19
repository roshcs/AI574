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
    # Expand abbreviations — only first occurrence per abbreviation
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
_ATOMIC_DOMAINS = frozenset({"recipe"})

# Separator ordering per atomic domain — used only when the document overflows.
_ATOMIC_SPLIT_SEPARATORS: Dict[str, list] = {
    # Prefer splitting between steps (`\n  N.`) and sections (`\n\n`) before
    # falling back to sentence / whitespace boundaries.
    "recipe": ["\nSteps:\n", "\n\n", "\n  ", "\n", ". ", " "],
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

        for doc in documents:
            # Apply domain preprocessing if available
            text = doc.page_content
            if preprocessor:
                text = preprocessor(text)

            # For atomic domains, keep the whole document as one chunk when it
            # fits — preserves coherence of short recipes / abstracts / etc.
            if is_atomic and len(text) <= self._char_chunk_size:
                text_chunks = [text]
                atomic_kept += 1
            elif is_atomic:
                text_chunks = atomic_splitter.split_text(text)
                atomic_split += 1
            else:
                text_chunks = self._splitter.split_text(text)

            parent_id = doc.metadata.get("id", doc.metadata.get("source", "unknown"))

            for i, chunk_text in enumerate(text_chunks):
                chunk_metadata = {
                    **doc.metadata,
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "parent_id": parent_id,
                    # For single-chunk atomic docs, keep the stable parent id —
                    # makes Chroma upserts idempotent across re-runs.
                    "id": parent_id if len(text_chunks) == 1 else f"{parent_id}_chunk_{i}",
                    "domain": domain,
                }
                all_chunks.append(
                    Document(page_content=chunk_text, metadata=chunk_metadata)
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
        preprocessor = _PREPROCESSORS.get(domain)
        if preprocessor:
            text = preprocessor(text)
        return self._splitter.split_text(text)
