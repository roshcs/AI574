"""
Document Loader
===============
Unified loader that normalizes different source formats into LangChain Documents.
Supports: PDF, text/markdown, CSV (Food.com), ArXiv API, and raw strings.

Usage:
    from ingestion.document_loader import DocumentLoader
    loader = DocumentLoader()
    docs = loader.load_pdf("manual.pdf")
    docs = loader.load_food_csv("RAW_recipes.csv", max_rows=10000)
    docs = loader.load_arxiv("transformer attention mechanisms", max_results=5)
"""

from __future__ import annotations

import ast
import csv
import hashlib
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

# Food.com recipe CSV fields carry Python-literal lists that can exceed the
# default csv field size limit on large rows — raise it once at import.
csv.field_size_limit(sys.maxsize)

logger = logging.getLogger(__name__)

try:
    import arxiv
except ImportError:  # pragma: no cover - exercised via runtime environments
    arxiv = None


_ALLOWED_EXTENSIONS = frozenset({".pdf", ".txt", ".md", ".csv"})
_MAX_LITERAL_EVAL_LEN = 100_000


def _validate_path(path: str, *, allowed_root: Optional[str] = None) -> Path:
    """Resolve *path* and reject traversal outside *allowed_root*."""
    resolved = Path(path).resolve()
    if allowed_root is not None:
        root = Path(allowed_root).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            raise ValueError(
                f"Path '{resolved}' is outside the allowed root '{root}'"
            )
    if resolved.suffix.lower() not in _ALLOWED_EXTENSIONS and resolved.is_file():
        raise ValueError(
            f"Unsupported file extension '{resolved.suffix}'. "
            f"Allowed: {sorted(_ALLOWED_EXTENSIONS)}"
        )
    return resolved


class DocumentLoader:
    """Unified document loader producing standardized LangChain Documents.

    Parameters
    ----------
    allowed_root : str or None
        If set, every file-loading method will reject paths that resolve
        outside this directory (path-traversal guard).
    """

    def __init__(self, allowed_root: Optional[str] = None):
        self.allowed_root = allowed_root

    # ── PDF / Text Files ──────────────────────────────────────────────────

    def load_pdf(self, path: str, source_label: Optional[str] = None) -> List[Document]:
        """Load a PDF file using PyPDF2."""
        from PyPDF2 import PdfReader

        validated = _validate_path(path, allowed_root=self.allowed_root)
        reader = PdfReader(str(validated))
        docs = []
        label = source_label or validated.stem

        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append(Document(
                    page_content=text.strip(),
                    metadata={
                        "source": label,
                        "file_path": path,
                        "page": i + 1,
                        "type": "pdf",
                    },
                ))
        logger.info(f"Loaded {len(docs)} pages from PDF: {path}")
        return docs

    def load_text(self, path: str, source_label: Optional[str] = None) -> List[Document]:
        """Load a plain text or markdown file."""
        validated = _validate_path(path, allowed_root=self.allowed_root)
        with open(validated, "r", encoding="utf-8") as f:
            content = f.read()
        label = source_label or validated.stem
        return [Document(
            page_content=content,
            metadata={
                "source": label,
                "file_path": path,
                "type": "text",
            },
        )]

    def load_directory(
        self,
        dir_path: str,
        extensions: tuple = (".pdf", ".txt", ".md"),
    ) -> List[Document]:
        """Load all supported files from a directory."""
        validated_dir = _validate_path(dir_path, allowed_root=self.allowed_root)
        docs = []
        for root, _, files in os.walk(str(validated_dir)):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                ext = Path(fname).suffix.lower()
                if ext == ".pdf":
                    docs.extend(self.load_pdf(fpath))
                elif ext in (".txt", ".md"):
                    docs.extend(self.load_text(fpath))
        logger.info(f"Loaded {len(docs)} documents from directory: {dir_path}")
        return docs

    # ── Food.com CSV ──────────────────────────────────────────────────────

    def load_food_csv(
        self,
        path: str,
        max_rows: Optional[int] = None,
        *,
        dedupe: bool = True,
        drop_junk: bool = True,
        interactions_csv: Optional[str] = None,
    ) -> List[Document]:
        """
        Load Food.com RAW_recipes.csv into Documents.

        Each row becomes one Document with structured recipe content and
        rich, filter-friendly metadata.

        Parameters
        ----------
        path : str
            Path to RAW_recipes.csv.
        max_rows : int, optional
            Stop after processing this many *input* rows. Dedupe and junk
            filtering still apply within the subset.
        dedupe : bool
            If True, drop exact duplicates based on
            hash(normalized_name, sorted(ingredients)). Earliest row wins.
        drop_junk : bool
            If True, filter rows with empty steps, missing ingredients, or
            implausible ``minutes`` values.
        interactions_csv : str, optional
            Path to RAW_interactions.csv. If provided, aggregate per-recipe
            ``review_count`` and ``avg_rating`` and attach to metadata.
        """
        validated = _validate_path(path, allowed_root=self.allowed_root)

        popularity = (
            _aggregate_interactions(interactions_csv, allowed_root=self.allowed_root)
            if interactions_csv
            else {}
        )
        if interactions_csv:
            logger.info(
                f"Loaded popularity signal for {len(popularity):,} recipes "
                f"from {interactions_csv}"
            )

        seen_hashes: set = set()
        dropped_junk = 0
        dropped_dupe = 0
        docs: List[Document] = []

        with open(validated, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_rows and i >= max_rows:
                    break

                if drop_junk and _is_junk_row(row):
                    dropped_junk += 1
                    continue

                if dedupe:
                    fp = _row_fingerprint(row)
                    if fp in seen_hashes:
                        dropped_dupe += 1
                        continue
                    seen_hashes.add(fp)

                metadata = _build_recipe_metadata(row, popularity)
                content = _format_recipe(row, metadata)
                if not content.strip():
                    continue

                docs.append(Document(page_content=content, metadata=metadata))

        logger.info(
            f"Loaded {len(docs):,} recipes from {path} "
            f"(dropped_junk={dropped_junk:,}, dropped_dupe={dropped_dupe:,})"
        )
        return docs

    # ── ArXiv API ─────────────────────────────────────────────────────────

    _ARXIV_MAX_RESULTS_CAP = 25

    @staticmethod
    def load_arxiv(
        query: str,
        max_results: int = 5,
        sort_by: str = "relevance",
    ) -> List[Document]:
        """
        Fetch papers from ArXiv API.
        Returns Documents with abstract + metadata (no full text by default).
        """
        if arxiv is None:
            raise ImportError(
                "arxiv package is required for load_arxiv(); install with `pip install arxiv`."
            )

        max_results = min(max_results, DocumentLoader._ARXIV_MAX_RESULTS_CAP)

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=(
                arxiv.SortCriterion.Relevance
                if sort_by == "relevance"
                else arxiv.SortCriterion.SubmittedDate
            ),
        )

        docs = []
        for paper in search.results():
            content = (
                f"Title: {paper.title}\n\n"
                f"Authors: {', '.join(a.name for a in paper.authors)}\n\n"
                f"Abstract: {paper.summary}\n\n"
                f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"ArXiv ID: {paper.entry_id}"
            )
            docs.append(Document(
                page_content=content,
                metadata={
                    "source": "arxiv",
                    "arxiv_id": paper.entry_id,
                    "title": paper.title,
                    "authors": [a.name for a in paper.authors],
                    "published": paper.published.isoformat(),
                    "pdf_url": paper.pdf_url,
                    "type": "scientific",
                    "id": f"arxiv_{paper.entry_id.split('/')[-1]}",
                },
            ))

        logger.info(f"Fetched {len(docs)} papers from ArXiv for: '{query}'")
        return docs

    # ── Raw Text Helper ───────────────────────────────────────────────────

    @staticmethod
    def from_texts(
        texts: List[str],
        domain: str,
        source: str = "manual",
    ) -> List[Document]:
        """Create Documents from raw text strings."""
        return [
            Document(
                page_content=t,
                metadata={
                    "source": source,
                    "type": domain,
                    "id": f"{domain}_{i}",
                },
            )
            for i, t in enumerate(texts)
        ]


# ── Private Helpers — Recipe Parsing & Enrichment ─────────────────────────────

# Food.com nutrition list = [calories, total_fat_pdv, sugar_pdv, sodium_pdv,
#   protein_pdv, saturated_fat_pdv, carbs_pdv]  (PDV = Percent Daily Value)
_NUTRITION_KEYS: Tuple[str, ...] = (
    "calories",
    "total_fat_pct",
    "sugar_pct",
    "sodium_pct",
    "protein_pct",
    "sat_fat_pct",
    "carbs_pct",
)

# Minimal tag → boolean flag mapping. Tags come lowercased from Food.com.
_DIETARY_FLAG_RULES: Dict[str, Tuple[str, ...]] = {
    "is_vegetarian": ("vegetarian",),
    "is_vegan": ("vegan",),
    "is_gluten_free": ("gluten-free", "gluten-free-diet"),
    "is_dairy_free": ("dairy-free",),
    "is_low_calorie": ("low-calorie", "low-in-something"),
}

_NAME_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def _safe_literal_list(raw: str) -> list:
    """Parse a Python-literal list field from the CSV; return [] on failure."""
    if not raw:
        return []
    if len(raw) > _MAX_LITERAL_EVAL_LEN:
        return []
    try:
        val = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    return val if isinstance(val, list) else []


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _minutes_bucket(minutes: int) -> str:
    if minutes <= 0:
        return "unknown"
    if minutes < 30:
        return "quick"
    if minutes <= 60:
        return "medium"
    return "long"


def _submitted_year(raw: str) -> int:
    if not raw:
        return 0
    # Food.com format: YYYY-MM-DD
    try:
        return int(raw[:4])
    except ValueError:
        return 0


def _is_junk_row(row: dict) -> bool:
    """Drop rows with empty steps, empty ingredients, or implausible prep time."""
    steps_raw = (row.get("steps") or "").strip()
    if not steps_raw or steps_raw == "[]":
        return True
    if not (row.get("ingredients") or "").strip():
        return True
    minutes = _safe_int(row.get("minutes"), default=0)
    # Reject <= 0 (missing) and > 1 day (junk like 1440000)
    if minutes <= 0 or minutes > 1440:
        return True
    return False


def _row_fingerprint(row: dict) -> str:
    """Exact-hash fingerprint over (normalized name, sorted ingredients)."""
    name = (row.get("name") or "").lower()
    name_norm = _NAME_NORMALIZE_RE.sub(" ", name).strip()
    ingredients = _safe_literal_list(row.get("ingredients") or "")
    ing_norm = "|".join(sorted(i.strip().lower() for i in ingredients if i))
    payload = f"{name_norm}::{ing_norm}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _parse_tags(raw: str) -> List[str]:
    tags = _safe_literal_list(raw)
    return [t.strip().lower() for t in tags if isinstance(t, str) and t.strip()]


def _parse_nutrition(raw: str) -> Dict[str, float]:
    values = _safe_literal_list(raw)
    result: Dict[str, float] = {}
    for key, val in zip(_NUTRITION_KEYS, values):
        result[key] = _safe_float(val)
    return result


def _derive_dietary_flags(tags: List[str]) -> Dict[str, bool]:
    tag_set = set(tags)
    return {
        flag: any(t in tag_set for t in rule_tags)
        for flag, rule_tags in _DIETARY_FLAG_RULES.items()
    }


def _popularity_tier(review_count: int, avg_rating: float) -> str:
    if review_count == 0:
        return "none"
    if review_count >= 50 and avg_rating >= 4.5:
        return "high"
    if review_count >= 10 and avg_rating >= 4.0:
        return "medium"
    return "low"


def _build_recipe_metadata(
    row: dict,
    popularity: Dict[str, Tuple[int, float]],
) -> Dict[str, object]:
    """Build the Chroma-compatible metadata dict for a single recipe row.

    Chroma metadata values must be str/int/float/bool (no lists or None).
    List-like fields (tags) are pipe-delimited strings with leading/trailing
    pipes so ``$contains`` filters can match exact tokens.
    """
    recipe_id = str(row.get("id", ""))
    minutes = _safe_int(row.get("minutes"))
    tags = _parse_tags(row.get("tags", ""))
    nutrition = _parse_nutrition(row.get("nutrition", ""))

    # |tag1|tag2|tag3|  — bookended so $contains('|vegan|') is an exact match
    tags_field = ("|" + "|".join(tags) + "|") if tags else ""

    meta: Dict[str, object] = {
        "source": "food.com",
        "type": "recipe",
        "domain": "recipe",
        "id": f"recipe_{recipe_id or row.get('name','')}",
        "recipe_id": recipe_id,
        "name": (row.get("name") or "").strip(),
        "minutes": minutes,
        "minutes_bucket": _minutes_bucket(minutes),
        "n_ingredients": _safe_int(row.get("n_ingredients")),
        "n_steps": _safe_int(row.get("n_steps")),
        "year": _submitted_year(row.get("submitted", "")),
        "tags": tags_field,
        "is_quick": minutes > 0 and minutes < 30,
    }
    meta.update(nutrition)
    meta.update(_derive_dietary_flags(tags))

    # Attach popularity signal if we aggregated interactions
    pop = popularity.get(recipe_id)
    if pop is not None:
        review_count, avg_rating = pop
        meta["review_count"] = review_count
        meta["avg_rating"] = round(avg_rating, 3)
        meta["popularity_tier"] = _popularity_tier(review_count, avg_rating)
    else:
        meta["review_count"] = 0
        meta["avg_rating"] = 0.0
        meta["popularity_tier"] = "none"

    return meta


def _format_recipe(row: dict, meta: Dict[str, object]) -> str:
    """Format a Food.com CSV row into readable recipe text.

    The text is what gets embedded, so we inline human-readable tags and
    nutrition to strengthen retrieval; structured metadata mirrors it for
    filtering.
    """
    parts: List[str] = []
    name = (row.get("name") or "Untitled Recipe").strip()
    parts.append(f"Recipe: {name}")

    minutes = meta.get("minutes", 0)
    if minutes:
        parts.append(f"Prep Time: {minutes} minutes ({meta.get('minutes_bucket')})")

    tags = _parse_tags(row.get("tags", ""))
    if tags:
        # Truncate to keep embeddings focused on salient tags
        parts.append("Tags: " + ", ".join(tags[:20]))

    desc = (row.get("description") or "").strip()
    if desc:
        parts.append(f"Description: {desc}")

    ingredients = _safe_literal_list(row.get("ingredients", ""))
    if ingredients:
        parts.append("Ingredients: " + ", ".join(ingredients))

    steps = _safe_literal_list(row.get("steps", ""))
    if steps:
        parts.append(
            "Steps:\n"
            + "\n".join(f"  {i+1}. {s}" for i, s in enumerate(steps))
        )

    calories = meta.get("calories", 0)
    if calories:
        parts.append(
            "Nutrition: "
            f"{calories:.0f} cal, "
            f"fat {meta.get('total_fat_pct', 0):.0f}%DV, "
            f"sugar {meta.get('sugar_pct', 0):.0f}%DV, "
            f"sodium {meta.get('sodium_pct', 0):.0f}%DV, "
            f"protein {meta.get('protein_pct', 0):.0f}%DV, "
            f"sat-fat {meta.get('sat_fat_pct', 0):.0f}%DV, "
            f"carbs {meta.get('carbs_pct', 0):.0f}%DV"
        )

    review_count = meta.get("review_count", 0)
    if review_count:
        parts.append(
            f"Popularity: {review_count} reviews, "
            f"avg rating {meta.get('avg_rating', 0):.2f} "
            f"({meta.get('popularity_tier')})"
        )

    return "\n".join(parts)


def _aggregate_interactions(
    path: str,
    allowed_root: Optional[str] = None,
) -> Dict[str, Tuple[int, float]]:
    """Aggregate RAW_interactions.csv into {recipe_id: (count, avg_rating)}.

    Streams the file — never materializes the full 1.1M-row table.
    """
    validated = _validate_path(path, allowed_root=allowed_root)
    counts: Dict[str, int] = {}
    sums: Dict[str, float] = {}

    with open(validated, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = (row.get("recipe_id") or "").strip()
            if not rid:
                continue
            rating = _safe_float(row.get("rating"), default=0.0)
            counts[rid] = counts.get(rid, 0) + 1
            sums[rid] = sums.get(rid, 0.0) + rating

    return {
        rid: (counts[rid], sums[rid] / counts[rid] if counts[rid] else 0.0)
        for rid in counts
    }
