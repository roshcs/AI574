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

import csv
import logging
import os
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Unified document loader producing standardized LangChain Documents."""

    # ── PDF / Text Files ──────────────────────────────────────────────────

    @staticmethod
    def load_pdf(path: str, source_label: Optional[str] = None) -> List[Document]:
        """Load a PDF file using PyPDF2."""
        from PyPDF2 import PdfReader

        reader = PdfReader(path)
        docs = []
        label = source_label or Path(path).stem

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

    @staticmethod
    def load_text(path: str, source_label: Optional[str] = None) -> List[Document]:
        """Load a plain text or markdown file."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        label = source_label or Path(path).stem
        return [Document(
            page_content=content,
            metadata={
                "source": label,
                "file_path": path,
                "type": "text",
            },
        )]

    @staticmethod
    def load_directory(
        dir_path: str,
        extensions: tuple = (".pdf", ".txt", ".md"),
    ) -> List[Document]:
        """Load all supported files from a directory."""
        docs = []
        for root, _, files in os.walk(dir_path):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                ext = Path(fname).suffix.lower()
                if ext == ".pdf":
                    docs.extend(DocumentLoader.load_pdf(fpath))
                elif ext in (".txt", ".md"):
                    docs.extend(DocumentLoader.load_text(fpath))
        logger.info(f"Loaded {len(docs)} documents from directory: {dir_path}")
        return docs

    # ── Food.com CSV ──────────────────────────────────────────────────────

    @staticmethod
    def load_food_csv(
        path: str,
        max_rows: Optional[int] = None,
    ) -> List[Document]:
        """
        Load Food.com RAW_recipes.csv into Documents.
        Each row becomes one Document with structured recipe content.
        """
        docs = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_rows and i >= max_rows:
                    break

                # Build readable recipe text
                content = _format_recipe(row)
                if not content.strip():
                    continue

                docs.append(Document(
                    page_content=content,
                    metadata={
                        "source": "food.com",
                        "recipe_id": row.get("id", str(i)),
                        "name": row.get("name", ""),
                        "minutes": row.get("minutes", ""),
                        "n_ingredients": row.get("n_ingredients", ""),
                        "type": "recipe",
                        "id": f"recipe_{row.get('id', i)}",
                    },
                ))

        logger.info(f"Loaded {len(docs)} recipes from: {path}")
        return docs

    # ── ArXiv API ─────────────────────────────────────────────────────────

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
        import arxiv

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


# ── Private Helpers ───────────────────────────────────────────────────────────

def _format_recipe(row: dict) -> str:
    """Format a Food.com CSV row into readable recipe text."""
    parts = []
    name = row.get("name", "Untitled Recipe")
    parts.append(f"Recipe: {name}")

    if row.get("minutes"):
        parts.append(f"Prep Time: {row['minutes']} minutes")

    if row.get("description"):
        parts.append(f"Description: {row['description']}")

    # Ingredients — stored as Python list string in CSV
    ingredients = row.get("ingredients", "")
    if ingredients:
        try:
            import ast
            ing_list = ast.literal_eval(ingredients)
            parts.append("Ingredients: " + ", ".join(ing_list))
        except (ValueError, SyntaxError):
            parts.append(f"Ingredients: {ingredients}")

    # Steps
    steps = row.get("steps", "")
    if steps:
        try:
            import ast
            step_list = ast.literal_eval(steps)
            parts.append("Steps:\n" + "\n".join(
                f"  {i+1}. {s}" for i, s in enumerate(step_list)
            ))
        except (ValueError, SyntaxError):
            parts.append(f"Steps: {steps}")

    if row.get("nutrition"):
        parts.append(f"Nutrition: {row['nutrition']}")

    return "\n".join(parts)
