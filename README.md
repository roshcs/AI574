# AI574: Multi-Domain CRAG Assistant

AI574 is a multi-agent retrieval-augmented system that routes user queries across three domains:
- `industrial`: PLC/SCADA troubleshooting and maintenance guidance
- `recipe`: cooking help, substitutions, and nutrition-aware responses
- `scientific`: paper summarization and ArXiv-backed research assistance

The system uses a supervisor router plus a shared Corrective RAG (CRAG) pipeline:
`retrieve -> grade -> rewrite (if needed) -> generate -> hallucination check`.

## Documentation

- [Documentation hub](docs/README.md): where to read first, repo map, notebooks, full test command
- [Contributing](CONTRIBUTING.md): venv, pre-commit (`nbstripout`), tests, secrets policy
- [Operations: indexing & Chroma](docs/OPERATIONS.md): offline index scripts and env vars
- [Architecture](ARCHITECTURE.md): design, diagrams, module responsibilities
- [Security](SECURITY.md): credentials and notebook hygiene

Architecture details and recommendations are documented in [ARCHITECTURE.md](./ARCHITECTURE.md).

## Architecture

- `orchestration/supervisor.py`: routes each query to one domain (or clarify/fallback)
- `orchestration/workflow_graph.py`: LangGraph workflow wiring
- `agents/`: domain specialists (industrial, recipe, scientific)
- `rag_core/`: shared CRAG components
- `foundation/`: LLM wrapper, embeddings, vector store
- `ingestion/`: loaders, chunking, indexing pipeline
- `evaluation/metrics.py`: routing and quality evaluation helpers

Routing flow:
`START -> supervisor -> {industrial|recipe|scientific|clarify|fallback} -> finalize -> END`

## Repository Layout

```text
AI574/
├── agents/
├── config/
├── evaluation/
├── foundation/
├── ingestion/
├── notebooks/
├── orchestration/
├── rag_core/
├── tests/
└── requirements.txt
```

## Requirements

- Python 3.10+
- Optional GPU for faster LLM + embedding inference
- Internet access when downloading model weights or querying ArXiv

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install "langchain-text-splitters>=0.2.0" "transformers>=4.45,<5.0"
```

If you want CUDA JAX backend (as used in notebooks):

```bash
pip install "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Configuration

Settings live in `config/settings.py` and can be overridden with environment variables:

- `EMBEDDING_MODEL` (default: `thenlper/gte-large`)
- `EMBEDDING_DEVICE` (`cuda` or `cpu`)
- `LLM_PRESET` (default: `gemma3_instruct_12b`)
- `LLM_BACKEND` (`jax`, `torch`, `tensorflow`)
- `LLM_MAX_TOKENS`, `LLM_SHORT_MAX_TOKENS`
- `CHROMA_PERSIST_DIR` (default: `./chroma_db`; the notebook sets this to `PROJECT_DIR/chroma_db` on Google Drive so the index persists across Colab sessions — full Food.com indexing can produce a large DB, avoid deleting it unless you intend to rebuild)
- `SEARCH_TOP_K`
- `DEFAULT_MODEL_ID` (default: `gemini_flash` — the LLM used by `run_query` when `model_id` is omitted)
- `GOOGLE_API_KEY` (required — used by the default `gemini_flash` model)
- `GROQ_API_KEY` (for Groq models via `model_id="groq_llama"`)
- `HOSTED_MODEL_FAILOVER_ID` (default: `gemma3`; used when hosted model calls fail due to auth/permission errors)

Example:

```bash
export KERAS_BACKEND=jax
export EMBEDDING_DEVICE=cpu
export CHROMA_PERSIST_DIR=./chroma_db
```

## Quickstart (Python)

```python
from config.settings import CONFIG
from foundation.llm_wrapper import KerasHubChatModel
from foundation.embedding_service import EmbeddingService
from foundation.vector_store import VectorStoreService
from ingestion.index_builder import IndexBuilder
from orchestration.workflow_graph import build_workflow, run_query

# 1) Core services
llm = KerasHubChatModel(config=CONFIG.llm)
embedder = EmbeddingService(config=CONFIG.embedding)
vector_store = VectorStoreService(embedding_service=embedder)
index_builder = IndexBuilder(vector_store=vector_store)

# 2) Index minimal demo data
index_builder.index_industrial_texts(
    [
        "Fault Code F0003 on Siemens S7-1200 indicates motor overtemperature. "
        "Check cooling fan, ambient temperature, and mechanical load."
    ],
    source="demo_manual",
)

# 3) Build workflow and query (uses gemini_flash by default)
workflow = build_workflow(llm=llm, vector_store=vector_store, index_builder=index_builder)
result = run_query(workflow, "My S7-1200 PLC shows fault F0003")

print(result["domain"])
print(result["confidence"])
print(result["model_id"])   # "gemini_flash" (the default)
print(result["response"])

# 4) Use local Gemma model explicitly
result = run_query(workflow, "My S7-1200 PLC shows fault F0003", model_id="gemma3")
print(result["response"])
```

## Runtime Model Selection

By default, `run_query` uses `gemini_flash` (Gemini 2.5 Flash) which requires
`GOOGLE_API_KEY` to be set. You can switch to any registered model at query time:

```python
# Default: uses gemini_flash (Gemini 2.5 Flash, requires GOOGLE_API_KEY)
result = run_query(workflow, "Fault F004 on my drive")

# Use local Gemma 3 12B (requires GPU + Kaggle credentials)
result = run_query(workflow, "Fault F004 on my drive", model_id="gemma3")

# Use Groq (Llama 3.1 8B) for fast inference (requires GROQ_API_KEY)
result = run_query(workflow, "Fault F004 on my drive", model_id="groq_llama")

# Combine with run modes
result = run_query(workflow, "Fault F004", model_id="gemini_flash", mode="fast_interactive")
```

If a hosted provider call fails with auth/permission errors (for example, `401`,
`403`, or revoked/leaked API keys), the hosted adapter automatically fails over to
`HOSTED_MODEL_FAILOVER_ID` (defaults to `gemma3`).

Available model IDs:

| `model_id` | Provider | Model | Requires |
|---|---|---|---|
| `gemini_flash` | Google | Gemini 2.5 Flash | `GOOGLE_API_KEY` (default) |
| `gemini_pro` | Google | Gemini 2.5 Pro | `GOOGLE_API_KEY` |
| `gemma3` | Local (KerasHub) | Gemma 3 12B | GPU + Kaggle credentials |
| `groq_llama` | Groq | Llama 3.1 8B Instant | `GROQ_API_KEY` |

To add custom models at runtime:

```python
from foundation.model_registry import register_model
register_model("my_model", lambda **kw: MyCustomLLM(**kw))
```

## Data Indexing

Use `ingestion/index_builder.py` helpers:

- Industrial manuals directory:
```python
index_builder.index_industrial_docs("./data/manuals")
```

- Food.com CSV (the notebook indexes the full CSV when `data/RAW_recipes.csv` exists under `PROJECT_DIR`; set `RECIPE_MAX_ROWS` to cap rows for debugging):
```python
index_builder.index_recipes("./data/RAW_recipes.csv", max_rows=None)  # all rows
```

- ArXiv (scientific):
```python
index_builder.index_arxiv_papers("transformer attention mechanism", max_results=10)
```

## Notebooks

- `notebooks/main_demo.ipynb`: compact end-to-end walkthrough
- `notebooks/Multi_Domain_Agent.ipynb`: fuller build + validation flow

These notebooks include installation cells and sample indexing blocks for each domain.

## Tests

Run unit tests:

```bash
python -m unittest tests/test_crag_pipeline.py
```

The current test suite uses mocks, so it does not require GPU or live model inference.

## Evaluation

`evaluation/metrics.py` provides:
- routing accuracy checks
- LLM-judge response scoring
- reliability checks (JSON parsing, confidence clamping)
- performance summaries (p50/p95 latency)

## Current Limitations

- Fallback route currently returns a static web-search recommendation message.
- Scientific retrieval uses ArXiv metadata/abstract content (not full PDF parsing by default).
- First-time model loading can take significant time due to weight downloads.

## Extending the System

To add a new domain:
1. Create a new agent in `agents/`.
2. Add prompt template in `config/prompts.py`.
3. Register domain in `config/settings.py` (`SupervisorConfig.domain_registry` + collections).
4. Index domain data through `IndexBuilder`/`VectorStoreService`.
