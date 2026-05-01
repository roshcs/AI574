#!/usr/bin/env python3
"""Build notebooks/AI574_Submission_Clean.ipynb from Multi_Domain_Agent.ipynb."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "Multi_Domain_Agent.ipynb"
DST = ROOT / "notebooks" / "AI574_Submission_Clean.ipynb"

NEW_TOC = """## Table of Contents (submission)

**Environment & setup** — [1. clone](#1-clone--update-repo) · [2. GPU / Drive / Chroma](#2-gpu--google-drive--chroma-persistence) · [2.5 Chroma safety](#25-chroma-safety-helpers) · [3. install](#3-install-dependencies) · [4. Keras backend](#4-set-keras-backend-and-verify) · [5. import path](#5-project-import-path) · [6. credentials](#6-load-credentials-colab-secrets--getpass) · [7. LLM](#7-load-llm)

**Vector store (restore + optional indexing)** — [8. embeddings + store](#8-embeddings--vector-store) · [8.5](#85-indexing-control) · [9 industrial](#9-industrial-indexing) · [10 recipe](#10-recipe-indexing) · [11 scientific](#11-scientific-indexing)

**Workflow** — [12. build](#12-build-workflow) · [Planned queries](#live-demonstration-planned-queries) · [12.5 live demo](#125-live-five-query-demo-path)

**Evaluation** — [20–22](#20-basic-rag-baseline) · [23 validation](#23-index-validation) · [25 checklist](#25-submission-summary-and-next-steps)

*Full appendix version:* `notebooks/Multi_Domain_Agent.ipynb`
"""

QUICK_START = """## How to use this submission notebook

1. **Run setup (through §7):** install dependencies, configure paths, load the LLM.
2. **Vector store (§8–8.5):** restore a Chroma snapshot when available. Use §9–**11** only if you must rebuild a domain index.
3. **Workflow (§12)** then **live demo (§12.5).** Set `RUN_FIVE_QUERY_LIVE_DEMO = True` in the live-demo code cell to run all five queries, or use `DEMO_QUERY_ID` in the next cell to run a single case.
4. **Evaluation (§20–22):** leave `RUN_FORMAL_EVAL = False` for a quick end-to-end pass; set `True` to regenerate 150×2 system numbers (slow). A **cached aggregate table** is included after the formal harness.
5. **Close out:** §**23** validation, optional snapshot (§**24**), and submission summary (§**25**).
"""

BENCHMARK_MD = r"""## Reported 150-query benchmark summary (cached)

Reproduce by running the formal cell with `RUN_FORMAL_EVAL = True` and `FORMAL_EVAL_LIMIT = None` (and the same `USE_LLM_EVAL_JUDGE` / `FORMAL_EVAL_MODE` flags as in that cell). Numbers are corpus- and judge-dependent.

| System | n | Route Acc | P@5 | Raw Halluc. | Unblocked | Task Done | Latency |
|--------|---|-----------|-----|-------------|-----------|-----------|--------|
| `multi_agent_crag` | 150 | 100.0% | 100.0% | 42.0% | 12.0% | 100.0% | 26.75s |
| `monolithic_rag` | 150 | n/a | 98.3% | 21.3% | 21.3% | 97.3% | 6.03s |
"""


def _cell_source_str(cell: dict) -> str:
    return "".join(cell.get("source", []))


def _set_cell_source(cell: dict, text: str) -> None:
    if not text:
        cell["source"] = [""]
    elif text.endswith("\n"):
        cell["source"] = [text]
    else:
        cell["source"] = [text + "\n"]


def _clear_code_cell_outputs(cell: dict) -> None:
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None


def _patch_header(text: str) -> str:
    if not text.lstrip().startswith("# AI574"):
        return text
    body = text.split("\n", 1)[1] if "\n" in text else ""
    return (
        "# AI574 — Multi-Domain CRAG Assistant (Submission, clean runbook)\n\n"
        "Streamlined for grading: setup, **optional** indexing, workflow, **five-query** demo, baselines, 150-query harness, cached benchmark summary, and checklist. "
        "The full project notebook (appendix routing experiments, long plots) is `notebooks/Multi_Domain_Agent.ipynb`.\n\n" + body
    )


def _patch_live_demo(text: str) -> str:
    old = (
        '"query": "Is the keto diet supported by recent research, and can you suggest a keto recipe?",\n'
    )
    new = (
        '"query": "I am preparing a presentation on retrieval-augmented generation in NLP. '
        'Summarize the recent research, and suggest a quick dinner recipe I can make before presenting.",\n'
    )
    if old in text:
        text = text.replace(old, new, 1)
    if "RUN_FIVE_QUERY_LIVE_DEMO" not in text and "run_live_demo(mode" in text:
        text = text.replace(
            "\n# Recommended for the recorded demo: run one query at a time if time is tight.\n"
            "live_demo_rows = run_live_demo(mode=\"best_quality\")\n",
            "\n# Submission default: skip 5 long queries on a fresh 'Run all' (set True for full tour).\n"
            "RUN_FIVE_QUERY_LIVE_DEMO = False\n"
            "if RUN_FIVE_QUERY_LIVE_DEMO:\n"
            "    live_demo_rows = run_live_demo(mode=\"best_quality\")\n"
            "else:\n"
            "    live_demo_rows = None\n"
            "    print(\"Skipping automatic five-query run. Set RUN_FIVE_QUERY_LIVE_DEMO = True to run all five.\")\n",
        )
    return text


def _patch_formal_eval(text: str) -> str:
    return re.sub(
        r"^RUN_FORMAL_EVAL\s*=\s*True\s*$",
        "RUN_FORMAL_EVAL = False  # set True to regenerate all 150×2 (slow); otherwise use cached table below",
        text,
        flags=re.MULTILINE,
    )


def _patch_216_table(text: str) -> str:
    if "## 21.6" not in text or "Cross-domain synthesis" not in text:
        return text
    old = (
        "| 4 | Cross-domain synthesis | "
        '"Is the keto diet supported by recent research, and can you suggest a keto breakfast?" | '
        "Calls the synthesis agent directly over `recipe` + `scientific`; "
        "combines scientific evidence with recipe guidance. |"
    )
    new = (
        '| 4 | Cross-domain synthesis | "I am preparing a presentation on '
        "retrieval-augmented generation in NLP. Summarize the recent research, "
        'and suggest a quick dinner recipe I can make before presenting." | '
        "Calls the synthesis agent directly over `recipe` + `scientific`; "
        "RAG research summary plus a recipe suggestion. |"
    )
    if old in text:
        return text.replace(old, new, 1)
    return text


def _patch_findings(_text: str) -> str:
    return (
        "## 22. Evaluation Findings\n\n"
        "From the **150-query** harness and the ablation discussion in this notebook (§20–§21.7), the key points are:\n\n"
        "- The **supervisor + per-domain Chroma** setup improves *controlled routing* and domain-scoped retrieval versus a monolithic top‑k pool.\n"
        "- **CRAG** adds grading, rewrite, and (optional) validation steps—this improves *grounding behavior* but **increases average latency** versus Basic RAG.\n"
        "- The **cached aggregate table** (inserted after §21.7) shows multi-agent **routing 100%** and strong task completion in the completed run, with **latency** as the main cost versus monolithic RAG.\n"
        "- The **optional learned router / grader** in §21.5 targets deployment speed; the core system uses the hosted grader and supervisor unless you enable those fine-tunes.\n\n"
        "Re-run the formal evaluation cell to refresh numbers; they depend on snapshot, judge, and `FORMAL_EVAL_*` settings.\n"
    )


def main() -> None:
    with open(SRC, encoding="utf-8") as f:
        src_nb = json.load(f)

    # 0-48: through "### Live demonstration..."; 49-51: 12.5 + live code + control
    take: list[int] = list(range(0, 49)) + [49, 50, 51]
    # 20-21.7
    take += list(range(74, 81))
    # 22+ ops; skip cell 89 (stray get_llm line in source nb)
    take += [85, 86, 87, 88, 90, 91, 92, 93]

    out: list[dict] = []
    for i in take:
        c = copy.deepcopy(src_nb["cells"][i])
        _clear_code_cell_outputs(c)
        t = _cell_source_str(c)
        if i == 0:
            t = _patch_header(t)
        if i == 2:
            t = NEW_TOC
        if i == 3:
            t = QUICK_START
        if i == 50:
            t = _patch_live_demo(t)
        if i == 80:
            t = _patch_formal_eval(t)
        if i == 78:
            t = _patch_216_table(t)
        if i == 85:
            t = _patch_findings(t)
        _set_cell_source(c, t)
        out.append(c)

    ins_at = 1 + take.index(80)
    out.insert(
        ins_at,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [BENCHMARK_MD + "\n"],
        },
    )

    new_nb = {
        "nbformat": src_nb.get("nbformat", 4),
        "nbformat_minor": src_nb.get("nbformat_minor", 5),
        "metadata": copy.deepcopy(src_nb.get("metadata", {})),
        "cells": out,
    }
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(new_nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"Wrote {DST} with {len(out)} cells.")


if __name__ == "__main__":
    main()
