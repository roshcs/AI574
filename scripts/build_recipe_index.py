#!/usr/bin/env python3
"""
Build the full recipe vector index from the Food.com corpus.

Runnable entry point that wires up EmbeddingService, VectorStoreService, and
IndexBuilder, then streams RAW_recipes.csv (with optional RAW_interactions.csv
popularity enrichment) into the ``recipe`` Chroma collection.

Example
-------
    # Full corpus with interactions + exact-hash dedupe (default)
    python scripts/build_recipe_index.py \\
        --data-dir ./data/recipe \\
        --batch-size 1000

    # Smoke test on first 1K rows without interactions
    python scripts/build_recipe_index.py \\
        --data-dir ./data/recipe \\
        --max-rows 1000 \\
        --no-interactions

    # Fresh rebuild (clears the recipe collection first)
    python scripts/build_recipe_index.py \\
        --data-dir ./data/recipe \\
        --clear

Environment
-----------
Respects the same env vars as the rest of AI574: ``EMBEDDING_MODEL``,
``CHROMA_PERSIST_DIR``. See ``config/settings.py`` for defaults.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Allow running from the project root without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from foundation.embedding_service import EmbeddingService  # noqa: E402
from foundation.vector_store import VectorStoreService  # noqa: E402
from ingestion.index_builder import IndexBuilder  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data-dir",
        default="./data/recipe",
        help="Directory containing RAW_recipes.csv and RAW_interactions.csv",
    )
    p.add_argument("--recipes-file", default="RAW_recipes.csv")
    p.add_argument("--interactions-file", default="RAW_interactions.csv")
    p.add_argument(
        "--no-interactions",
        action="store_true",
        help="Skip the popularity-signal aggregation from RAW_interactions.csv",
    )
    p.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable exact-hash dedupe (keeps every row)",
    )
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap input rows (useful for smoke tests; default: all ~231K)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Recipes per embed/upsert batch (default: 1000)",
    )
    p.add_argument(
        "--clear",
        action="store_true",
        help="Delete the existing recipe collection before indexing",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    log = logging.getLogger("build_recipe_index")

    data_dir = Path(args.data_dir).resolve()
    recipes_path = data_dir / args.recipes_file
    interactions_path = data_dir / args.interactions_file

    if not recipes_path.is_file():
        log.error("Recipes CSV not found: %s", recipes_path)
        return 2

    if args.no_interactions:
        interactions_arg = None
    elif not interactions_path.is_file():
        log.warning(
            "Interactions CSV not found (%s); proceeding without popularity enrichment",
            interactions_path,
        )
        interactions_arg = None
    else:
        interactions_arg = str(interactions_path)

    # Build the service stack
    log.info("Instantiating embedding service: %s", os.getenv("EMBEDDING_MODEL", "default"))
    embedder = EmbeddingService()
    vector_store = VectorStoreService(embedding_service=embedder)

    if args.clear:
        log.warning("--clear: wiping the 'recipe' collection")
        vector_store.clear_collection("recipe")

    builder = IndexBuilder(vector_store=vector_store)

    t0 = time.time()
    count = builder.index_recipes(
        str(recipes_path),
        max_rows=args.max_rows,
        dedupe=not args.no_dedupe,
        interactions_csv=interactions_arg,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - t0

    stats = vector_store.get_collection_stats("recipe")
    log.info(
        "Done: indexed %d chunks in %.1fs. Collection '%s' now has %d documents.",
        count,
        elapsed,
        stats["collection_name"],
        stats["document_count"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
