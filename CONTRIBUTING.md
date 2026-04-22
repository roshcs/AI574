# Contributing to AI574

## Setup

1. Clone the repository and create a Python 3.10+ virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -U pip
   pip install -r requirements.txt
   ```

2. Optional stacks (JAX/CUDA, extra splitters, hosted providers) are documented in [README.md](./README.md).

## Pre-commit hooks

This repo uses [`.pre-commit-config.yaml`](./.pre-commit-config.yaml) with **nbstripout** so notebook outputs and Colab metadata are stripped before commit—reducing accidental leakage of secrets in notebook output.

If you use pre-commit locally:

```bash
pip install pre-commit
pre-commit install
```

Hooks run on `git commit`; you can also run `pre-commit run --all-files` once.

## Tests

Tests use mocks and do **not** require a GPU or live API calls.

Run the full suite from the repo root:

```bash
python -m unittest discover -s tests
```

Modules:

- `tests/test_crag_pipeline.py` — CRAG pipeline behavior
- `tests/test_security.py` — security-related checks

## Secrets and notebooks

Never commit API keys, Kaggle credentials, or tokens in `.py`, `.ipynb`, `.md`, or YAML files tracked by git.

- Local: use a `.env` file (ignored by git) or your shell environment.
- Colab: prefer `userdata` or `getpass`; patterns are described in [SECURITY.md](./SECURITY.md).

If credentials were exposed, treat them as compromised and rotate them following [SECURITY.md](./SECURITY.md).

## Documentation

- Hub: [docs/README.md](./docs/README.md)
- Architecture: [ARCHITECTURE.md](./ARCHITECTURE.md)
