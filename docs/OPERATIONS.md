# Operations: indexing and vector storage

This page summarizes **offline indexing** workflows. CLI details and flags live in each script’s docstring and `--help` output.

## Environment variables

Indexing uses the same embedding and Chroma settings as the rest of the app. Defaults and overrides are defined in [`config/settings.py`](../config/settings.py).

| Variable | Default (if unset) | Effect on indexing |
|----------|-------------------|---------------------|
| `EMBEDDING_MODEL` | `thenlper/gte-large` | Embedding model used for all chunks |
| `EMBEDDING_DEVICE` | `cuda` | CPU vs GPU for embeddings (`cpu` is fine for smaller jobs) |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | On-disk Chroma persistence; reuse the same path across sessions so you do not rebuild unnecessarily |

Other variables (`SEARCH_TOP_K`, LLM-related vars) matter at **query** time more than during indexing; see [README.md](../README.md).

## When to use scripts vs notebooks

| Situation | Approach |
|-----------|----------|
| Quick demos or mixed-domain experiments | [`IndexBuilder`](../ingestion/index_builder.py) from Python or [notebooks/main_demo.ipynb](../notebooks/main_demo.ipynb) / [Multi_Domain_Agent.ipynb](../notebooks/Multi_Domain_Agent.ipynb) |
| Full **Food.com** recipe corpus into the `recipe` collection | [`scripts/build_recipe_index.py`](../scripts/build_recipe_index.py) — streams CSV in batches, optional interactions file, dedupe and `--clear` for rebuild |
| Large **ArXiv** snapshot (e.g. Kaggle metadata JSON) for the `scientific` collection | [`scripts/build_scientific_index.py`](../scripts/build_scientific_index.py) — filters by category/year, dedupes on `arxiv_id`, skips IDs already in Chroma so runs can resume |

Small industrial PDF/text batches can stay on `index_builder.index_industrial_docs(...)` as in the README quickstart.

## Recipe index (`build_recipe_index.py`)

- Points `--data-dir` at a folder containing `RAW_recipes.csv` (and optionally `RAW_interactions.csv`).
- Example invocations (full corpus, smoke test, fresh rebuild) are in the script header in [`scripts/build_recipe_index.py`](../scripts/build_recipe_index.py).

## Scientific index (`build_scientific_index.py`)

- Consumes an ArXiv OAI metadata snapshot (e.g. Cornell/Kaggle dump), not full PDFs by default.
- Defaults emphasize CS-oriented categories and recent years; you can widen `--categories`.
- Dedupes on `arxiv_id` and skips documents already present in Chroma so interrupted runs are cheap to continue.
- The scientific agent can still query the live ArXiv API at runtime for gaps; see script comments in [`scripts/build_scientific_index.py`](../scripts/build_scientific_index.py).

## Persistence and disk usage

Full recipe indexing can produce a **large** Chroma database. Keep `CHROMA_PERSIST_DIR` stable (for example on Google Drive in Colab) if you rely on long-lived indexes—avoid deleting it unless you intend to rebuild.
