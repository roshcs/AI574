# AI574 documentation

Start here if you want a single map of the repository. Deep dives stay in the linked pages so we avoid duplicating [ARCHITECTURE.md](../ARCHITECTURE.md).

## Choose your path

| Goal | Where to go |
|------|-------------|
| Install, configure env vars, run `run_query`, switch models | [README.md](../README.md) |
| Layered design, CRAG flow, module responsibilities | [ARCHITECTURE.md](../ARCHITECTURE.md) (see §5 for module breakdown) |
| Credential hygiene, notebook safety, rotation | [SECURITY.md](../SECURITY.md) |
| Venv, tests, pre-commit, contributing | [CONTRIBUTING.md](../CONTRIBUTING.md) |
| Offline indexing scripts, Chroma persistence | [OPERATIONS.md](./OPERATIONS.md) |

## Repository map

| Directory | Role |
|-----------|------|
| `config/` | Prompts and typed settings (`settings.py`, env overrides) |
| `foundation/` | LLM adapter, embeddings, Chroma vector store |
| `ingestion/` | Loaders, chunking, `IndexBuilder` |
| `rag_core/` | CRAG pipeline: retriever, grader, rewriter, generator, hallucination check |
| `agents/` | Domain agents (`Industrial`, `Recipe`, `Scientific`) on top of CRAG |
| `orchestration/` | Supervisor routing + LangGraph workflow (`workflow_graph.py`) |
| `evaluation/` | Routing and response quality helpers |
| `scripts/` | CLI jobs for large recipe/scientific indexes |
| `tests/` | Unit tests (mocked; no GPU required) |

For diagrams and end-to-end runtime flow, use [ARCHITECTURE.md](../ARCHITECTURE.md).

## Notebooks

| Notebook | Use when |
|----------|----------|
| [notebooks/main_demo.ipynb](../notebooks/main_demo.ipynb) | Short end-to-end walkthrough |
| [notebooks/Multi_Domain_Agent.ipynb](../notebooks/Multi_Domain_Agent.ipynb) | Fuller build, indexing, and validation |

Do not commit API keys in notebooks. Prefer environment variables or Colab `userdata`; see [SECURITY.md](../SECURITY.md).

## Testing

Run all tests from the repository root:

```bash
python -m unittest discover -s tests
```

This picks up `test_crag_pipeline` and `test_security`. Individual modules:

```bash
python -m unittest tests.test_crag_pipeline
python -m unittest tests.test_security
```

## CI note

The manual [Sync to Google Drive](../.github/workflows/sync-to-gdrive.yml) workflow runs a secret scan on tracked files. If your changes include patterns that look like API keys, the sync step may fail until those strings are removed or redacted.
