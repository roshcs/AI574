#!/usr/bin/env python3
"""
Build the scientific vector index from the Cornell-University/arxiv Kaggle dump.

Defaults target computer-science papers from 2020 onward, which matches the
topics exercised by ``evaluation.metrics.SCIENTIFIC_TEST_QUERIES``.

Runtime behaviour
-----------------
* Filters on category prefixes (default ``cs.``) and year range (default 2020+).
* Dedupes on ``arxiv_id`` within a single run.
* Skips papers already in Chroma by stable ID, so interrupted runs resume cheaply.
* The ScientificAgent still backfills from the live ArXiv API at query time for
  anything the local index misses.

Example
-------
    # Full cs.* corpus from 2020 onward
    python scripts/build_scientific_index.py \\
        --dump ./data/scientific/arxiv-metadata-oai-snapshot.json

    # Smoke test on first 500 matching papers
    python scripts/build_scientific_index.py \\
        --dump ./data/scientific/arxiv-metadata-oai-snapshot.json \\
        --max-rows 500

    # Broader scope — include stat.ML and q-bio.*
    python scripts/build_scientific_index.py \\
        --dump ./data/scientific/arxiv-metadata-oai-snapshot.json \\
        --categories cs. stat.ML q-bio.

Environment
-----------
Respects ``EMBEDDING_MODEL`` and ``CHROMA_PERSIST_DIR``. See
``config/settings.py`` for defaults.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from foundation.embedding_service import EmbeddingService  # noqa: E402
from foundation.vector_store import VectorStoreService  # noqa: E402
from ingestion.index_builder import IndexBuilder  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dump",
        default="./data/scientific/arxiv-metadata-oai-snapshot.json",
        help="Path to arxiv-metadata-oai-snapshot.json (NDJSON)",
    )
    p.add_argument(
        "--categories",
        nargs="+",
        default=["cs."],
        help="Accept papers whose categories match ANY of these prefixes "
             "(default: cs.). Pass empty-string to disable the filter.",
    )
    p.add_argument("--year-min", type=int, default=2020)
    p.add_argument("--year-max", type=int, default=None)
    p.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap kept papers (useful for smoke tests)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Papers per embed + upsert round-trip (default: 1000)",
    )
    p.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Disable in-stream arxiv_id dedupe",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-embed every paper even if its arxiv_<id> is in Chroma",
    )
    p.add_argument(
        "--clear",
        action="store_true",
        help="Delete the 'scientific' collection before indexing",
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
    log = logging.getLogger("build_scientific_index")

    dump_path = Path(args.dump).resolve()
    if not dump_path.is_file():
        log.error("Arxiv dump not found: %s", dump_path)
        log.error(
            "Hint: download Cornell-University/arxiv via kagglehub and place "
            "the JSON file at data/scientific/arxiv-metadata-oai-snapshot.json."
        )
        return 2

    log.info("Instantiating embedding service: %s",
             os.getenv("EMBEDDING_MODEL", "default"))
    embedder = EmbeddingService()
    vector_store = VectorStoreService(embedding_service=embedder)

    if args.clear:
        log.warning("--clear: wiping the 'scientific' collection")
        vector_store.clear_collection("scientific")

    builder = IndexBuilder(vector_store=vector_store)

    categories = tuple(c for c in args.categories if c) or ()
    skip_existing = not args.force
    if args.force:
        log.info("--force: will re-embed all papers (skip_existing=False)")

    t0 = time.time()
    count = builder.index_arxiv_dump(
        str(dump_path),
        max_rows=args.max_rows,
        categories=categories,
        year_min=args.year_min,
        year_max=args.year_max,
        dedupe=not args.no_dedupe,
        batch_size=args.batch_size,
        skip_existing=skip_existing,
    )
    elapsed = time.time() - t0

    stats = vector_store.get_collection_stats("scientific")
    log.info(
        "Done: indexed %d new papers in %.1fs. Collection '%s' now has %d documents.",
        count,
        elapsed,
        stats["collection_name"],
        stats["document_count"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
