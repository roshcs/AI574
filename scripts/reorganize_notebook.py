"""
Reorganize and renumber the AI574 submission notebook.

Single-pass, idempotent script that:
  1. Drops pure-clutter cells (empty markdown/code, fully commented-out
     operational recovery scripts that don't belong in a submission).
  2. Demotes specific known duplicate H2 cells (the "## Runtime Model
     Selection" and "## Why does one query take 3+ minutes?" cells that
     immediately follow another H2 with the same topic) to H3 subsections.
  3. Reorders sections into the canonical "submission" flow:
        Setup -> Indexing -> Workflow & Demos -> Advanced Agents
        -> Evaluation -> Operational Validation -> Submission Summary.
  4. Renumbers every "## N." heading sequentially.
  5. Inserts a Table of Contents markdown cell right after the project
     framing cell.

Never strips outputs. Never edits code-cell sources.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "Multi_Domain_Agent.ipynb"


# ── Cell-level helpers ───────────────────────────────────────────────────────


def _cell_text(cell: dict) -> str:
    return "".join(cell.get("source", [])).strip()


def _first_line(cell: dict) -> str:
    text = _cell_text(cell)
    return text.splitlines()[0] if text else ""


COMMENTED_CLUTTER_PATTERNS = [
    # Cell 4 (chroma persist dir wipe helper)
    re.compile(r"^# import os, shutil[\s\S]*Cleared\. Re-run from", re.MULTILINE),
    # Cell 28 (sqlite integrity check helper)
    re.compile(r"^# import os, subprocess, sqlite3[\s\S]*PRAGMA integrity_check", re.MULTILINE),
    # Cell 29 (swap recovered db)
    re.compile(r"^# import os, shutil, datetime, sqlite3[\s\S]*recovered = db \+ \"\.recovered\"", re.MULTILINE),
    # Cell 67 (manual snapshot setup)
    re.compile(r"^# import os, tarfile, pathlib, datetime, sqlite3, gc[\s\S]*PROJECT_DIR", re.MULTILINE),
]


def _is_clutter(cell: dict) -> bool:
    text = _cell_text(cell)
    if not text:
        return True
    for pat in COMMENTED_CLUTTER_PATTERNS:
        if pat.search(text):
            return True
    return False


# ── H2 detection / matching against the HEADING LINE ONLY ────────────────────

H2_LINE_RE = re.compile(r"^##\s+(\d+(?:\.\d+)?)?\s*\.?\s*(.+)$")


def _is_h2_header(cell: dict) -> bool:
    if cell.get("cell_type") != "markdown":
        return False
    line = _first_line(cell)
    if not line.startswith("## "):
        return False
    if line.startswith("### "):
        return False
    return True


def _h2_heading_line(cell: dict) -> str:
    return _first_line(cell) if _is_h2_header(cell) else ""


# ── Targeted demotion of known duplicate H2 cells ────────────────────────────
# Only demote cells we know are duplicate sub-headings, not generic "two H2
# cells in a row". This keeps separate sections (e.g. Findings vs Comparison)
# from being collapsed.
DEMOTE_FIRST_LINES = {
    "## Runtime Model Selection",
    "## Why does one query take 3+ minutes? / LLM throughput on A100",
}


def _maybe_demote(cell: dict) -> dict:
    if cell.get("cell_type") != "markdown":
        return cell
    line = _first_line(cell)
    if line not in DEMOTE_FIRST_LINES:
        return cell
    src = _cell_text(cell)
    src = re.sub(r"^##\s+", "### ", src, count=1)
    new_cell = copy.deepcopy(cell)
    new_cell["source"] = [s + "\n" for s in src.splitlines()]
    return new_cell


# ── Canonical section plan ───────────────────────────────────────────────────
# Each entry = (canonical_number, canonical_title, matcher_substrings).
# `matcher_substrings` is matched against the *heading line only* of each H2
# cell (lowercased), not the whole markdown body.
SECTION_PLAN: List[Tuple[str, str, List[str]]] = [
    # Setup (1-7)
    ("1",   "Clone / update repo",                                 ["clone / update repo"]),
    ("2",   "GPU + Google Drive + Chroma persistence",             ["gpu + google drive"]),
    ("2.5", "Chroma safety helpers",                               ["chroma safety helpers"]),
    ("3",   "Install dependencies",                                ["install dependencies"]),
    ("4",   "Set Keras backend and verify",                        ["set keras backend"]),
    ("5",   "Project import path",                                 ["project import path"]),
    ("6",   "Load credentials (Colab Secrets / getpass)",          ["load credentials"]),
    ("7",   "Load LLM",                                            ["load llm"]),
    # Indexing (8-11)
    ("8",   "Embeddings + vector store",                           ["embeddings + vector store"]),
    ("8.5", "Indexing control",                                    ["indexing control"]),
    ("9",   "Industrial indexing",                                 ["industrial indexing"]),
    ("10",  "Recipe indexing",                                     ["recipe indexing"]),
    ("11",  "Scientific indexing",                                 ["scientific indexing"]),
    # Workflow & Demos (12-15)
    ("12",  "Build workflow",                                      ["build workflow"]),
    ("13",  "Per-domain query demos",                              ["query demos"]),
    ("14",  "Runtime model selection",                             ["runtime model selection"]),
    ("15",  "Latency analysis",                                    ["latency analysis", "performance notes"]),
    # Advanced Agents (16-19)
    ("16",  "Hybrid Router Demo",                                  ["hybrid router demo"]),
    ("17",  "Supervisor Routing Improvement Demo",                 ["supervisor routing improvement"]),
    ("18",  "Cross-Domain Synthesis Agent",                        ["cross-domain synthesis"]),
    ("19",  "Web-Search Fallback Agent",                           ["web-search fallback"]),
    # Evaluation (20-22)
    ("20",  "Basic RAG Baseline",                                  ["basic rag baseline"]),
    ("21",  "Baselines vs CRAG: Ablation Comparison",              ["baselines vs crag"]),
    ("22",  "Evaluation Findings",                                 ["evaluation findings"]),
    # Operational Validation (23-24)
    ("23",  "Index validation",                                    ["__inject_index_validation__"]),
    ("24",  "Final snapshot",                                      ["final snapshot"]),
    # Submission (25)
    ("25",  "Submission Summary and Next Steps",                   ["submission summary"]),
]


# ── Slicing helpers ──────────────────────────────────────────────────────────


def _find_section_index(plan_entry: Tuple[str, str, List[str]],
                        cells: List[dict],
                        skip_ids: set) -> int:
    """Return the cell index of the H2 marking this section, or -1.

    Only matches against the H2 heading line, not the cell body. Skips any
    cells whose `id()` is in `skip_ids`.
    """
    _, _, fragments = plan_entry
    fragments_lower = [f.lower() for f in fragments]
    for i, c in enumerate(cells):
        if id(c) in skip_ids:
            continue
        line = _h2_heading_line(c).lower()
        if not line:
            continue
        if any(frag in line for frag in fragments_lower):
            return i
    return -1


def _slice_section(cells: List[dict], start: int, skip_ids: set) -> List[dict]:
    """Return [start_header, ...cells until next H2_or_H1)] but skip any
    cells already consumed by a previous section.
    """
    if start < 0 or start >= len(cells):
        return []
    out = [cells[start]]
    j = start + 1
    while j < len(cells):
        c = cells[j]
        if id(c) in skip_ids:
            j += 1
            continue
        if _is_h2_header(c):
            break
        # H1 cells should also stop a section.
        if c.get("cell_type") == "markdown" and _first_line(c).startswith("# "):
            break
        out.append(c)
        j += 1
    return out


# ── Index validation section ─────────────────────────────────────────────────
# The original notebook puts two operational checks right before "## 16. Final
# snapshot": (a) a re-init of `vs` + `print(vs.get_all_stats())`, and (b) a
# detailed `EXPECTED = {...}` per-collection sanity check. We synthesise §22
# from those two cells, anchoring on `EXPECTED = {` which is unique.

ANCHOR_HINTS = ("EXPECTED = {", "min_sim", "overall_ok")


def _is_index_validation_anchor(cell: dict) -> bool:
    if cell.get("cell_type") != "code":
        return False
    text = _cell_text(cell)
    return all(h in text for h in ANCHOR_HINTS)


def _is_vs_stats_reinit(cell: dict) -> bool:
    """Match the small `from foundation.vector_store import VectorStoreService;
    vs = VectorStoreService(...); print(vs.get_all_stats())` cell that pairs
    with the §22 anchor. Distinguished from the original §8 init cell by the
    absence of the `"✅ Vector store initialized"` print.
    """
    if cell.get("cell_type") != "code":
        return False
    text = _cell_text(cell)
    return (
        "VectorStoreService(embedding_service=embedder)" in text
        and "vs.get_all_stats()" in text
        and "Vector store initialized" not in text
    )


def _slice_index_validation(cells: List[dict], skip_ids: set) -> List[dict]:
    """Locate the §22 block:
      * anchor on the unique `EXPECTED = { ... overall_ok ... min_sim` cell;
      * include the immediately-preceding `vs.get_all_stats()` reinit cell
        (only if it's that very specific shape — never any other code cell).
    """
    anchor_idx: Optional[int] = None
    for i, c in enumerate(cells):
        if id(c) in skip_ids:
            continue
        if _is_index_validation_anchor(c):
            anchor_idx = i
            break
    if anchor_idx is None:
        return []

    start_idx = anchor_idx
    j = anchor_idx - 1
    while j >= 0 and id(cells[j]) in skip_ids:
        j -= 1
    if j >= 0 and _is_vs_stats_reinit(cells[j]):
        start_idx = j

    return cells[start_idx : anchor_idx + 1]


def _make_markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.rstrip().splitlines()],
    }


def _renumber_h2_in_place(cell: dict, new_number: str, new_title: str) -> None:
    """Rewrite the first line of an H2 markdown cell in place."""
    if cell.get("cell_type") != "markdown":
        return
    src = _cell_text(cell)
    if not src:
        return
    lines = src.splitlines()
    if re.match(r"^##\s+", lines[0]):
        # Sub-section numbers like "8.5" already contain a dot; don't add a
        # second one. Only top-level integer sections get the trailing dot.
        sep = " " if "." in new_number else ". "
        lines[0] = f"## {new_number}{sep}{new_title}".replace("  ", " ")
        if "." not in new_number:
            lines[0] = f"## {new_number}. {new_title}"
    new_text = "\n".join(lines).rstrip() + "\n"
    cell["source"] = [s + "\n" for s in new_text.splitlines()]


def _build_toc(plan: List[Tuple[str, str, List[str]]]) -> dict:
    title_for = {num: title for num, title, _ in plan}
    groupings = [
        ("**Setup**",            ["1", "2", "2.5", "3", "4", "5", "6", "7"]),
        ("**Indexing**",         ["8", "8.5", "9", "10", "11"]),
        ("**Workflow & Demos**", ["12", "13", "14", "15"]),
        ("**Advanced Agents**",  ["16", "17", "18", "19"]),
        ("**Evaluation**",       ["20", "21", "22"]),
        ("**Operational**",      ["23", "24"]),
        ("**Submission**",       ["25"]),
    ]
    parts = ["## Table of Contents", ""]
    for label, nums in groupings:
        parts.append(label)
        for n in nums:
            if n in title_for:
                sep = " " if "." in n else ". "
                parts.append(f"- §{n}{sep}{title_for[n]}")
        parts.append("")
    text = "\n".join(parts).rstrip() + "\n"
    return _make_markdown_cell(text)


# ── Main reorganization ──────────────────────────────────────────────────────


def _is_synth_cell(cell: dict) -> bool:
    """Markdown cells that this script previously synthesised (TOC, §22
    intro header, Appendix marker). Drop them so reruns stay idempotent.
    """
    if cell.get("cell_type") != "markdown":
        return False
    line = _first_line(cell)
    if line == "## Table of Contents":
        return True
    text = _cell_text(cell)
    if "Final operational checks before submission" in text:
        return True
    if line in {"## 22. Index validation", "## 23. Index validation"}:
        return True
    if line.startswith("## Appendix: cells not classified"):
        return True
    return False


def reorganize(nb: dict) -> dict:
    cells = nb["cells"]

    # 1) drop clutter and previously-synthesised cells (idempotency)
    cleaned = [
        c for c in cells
        if not _is_clutter(c) and not _is_synth_cell(c)
    ]

    # 2) demote known duplicate H2 cells (specific titles only)
    cleaned = [_maybe_demote(c) for c in cleaned]

    # 3) preserve title + framing at the top
    head = cleaned[:2]
    body = cleaned[2:]

    # 4) PRE-PASS: claim the §23 (Index validation) cells *before* any
    # section slicing runs, so §15's slice doesn't absorb them. We re-emit
    # them at canonical position §23 in the main loop.
    consumed: set = set()
    index_validation_cells = _slice_index_validation(body, consumed)
    for c in index_validation_cells:
        consumed.add(id(c))

    # 5) walk the canonical plan, slicing each section out of body in order
    new_body: List[dict] = []

    for entry in SECTION_PLAN:
        number, title, _ = entry
        if number == "23":
            if index_validation_cells:
                header = _make_markdown_cell(
                    f"## {number}. {title}\n\n"
                    "Final operational checks before submission: confirms each "
                    "Chroma collection is non-empty, metadata fields are "
                    "populated, and semantic similarity probes return reasonable "
                    "top hits."
                )
                new_body.append(header)
                new_body.extend(index_validation_cells)
            continue

        idx = _find_section_index(entry, body, consumed)
        if idx < 0:
            continue
        section = _slice_section(body, idx, consumed)
        if not section:
            continue
        # Renumber the section header in place (mutates the original cell
        # object so id()-based dedup keeps working).
        _renumber_h2_in_place(section[0], number, title)
        for c in section:
            consumed.add(id(c))
        new_body.extend(section)

    # 6) collect leftovers (defensive - we want lossless behavior)
    leftovers = [
        c for c in body
        if id(c) not in consumed and not _is_clutter(c)
    ]
    if leftovers:
        new_body.append(_make_markdown_cell(
            "## Appendix: cells not classified by the reorganizer"
        ))
        new_body.extend(leftovers)

    # 7) prepend Table of Contents right after framing
    toc = _build_toc(SECTION_PLAN)
    nb["cells"] = head + [toc] + new_body
    return nb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-place", action="store_true",
                   help="Write back to the source notebook (default: write reorganized.ipynb).")
    p.add_argument("--out", default=None,
                   help="Output path (defaults to .reorganized.ipynb beside source).")
    args = p.parse_args()

    nb_path = NOTEBOOK_PATH
    with open(nb_path) as f:
        nb = json.load(f)

    before = len(nb["cells"])
    nb = reorganize(nb)
    after = len(nb["cells"])

    if args.in_place:
        out_path = nb_path
    else:
        out_path = Path(args.out) if args.out else nb_path.with_suffix(".reorganized.ipynb")

    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")

    print(f"cells: {before} -> {after}")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
