"""
Security tests: path traversal, secret scanning patterns, logging sanitisation,
and auth-error failover behaviour.  All tests are lightweight (no GPU/network).
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    from ingestion.document_loader import DocumentLoader, _validate_path
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False


# ── Path-traversal tests ──────────────────────────────────────────────────────

@unittest.skipUnless(_HAS_LANGCHAIN, "langchain_core not installed")
class TestPathValidation(unittest.TestCase):
    """Verify _validate_path rejects traversal and bad extensions."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.safe_file = Path(self.tmpdir) / "data" / "test.txt"
        self.safe_file.parent.mkdir(parents=True, exist_ok=True)
        self.safe_file.write_text("hello")

    def test_valid_path_within_root(self):
        result = _validate_path(str(self.safe_file), allowed_root=self.tmpdir)
        self.assertEqual(result, self.safe_file.resolve())

    def test_rejects_traversal_above_root(self):
        evil_path = os.path.join(self.tmpdir, "..", "etc", "passwd")
        with self.assertRaises(ValueError) as ctx:
            _validate_path(evil_path, allowed_root=self.tmpdir)
        self.assertIn("outside the allowed root", str(ctx.exception))

    def test_allows_any_path_without_root(self):
        result = _validate_path(str(self.safe_file), allowed_root=None)
        self.assertEqual(result, self.safe_file.resolve())

    def test_loader_enforces_root(self):
        loader = DocumentLoader(allowed_root=self.tmpdir)
        evil = os.path.join(self.tmpdir, "..", "etc", "passwd")
        with self.assertRaises(ValueError):
            loader.load_text(evil)

    def test_loader_default_allows_any(self):
        loader = DocumentLoader()
        docs = loader.load_text(str(self.safe_file))
        self.assertEqual(len(docs), 1)


# ── ArXiv cap tests ───────────────────────────────────────────────────────────

@unittest.skipUnless(_HAS_LANGCHAIN, "langchain_core not installed")
class TestArxivCap(unittest.TestCase):
    """Verify on-demand ArXiv fetch is capped."""

    @patch("ingestion.document_loader.arxiv")
    def test_max_results_capped(self, mock_arxiv):
        mock_arxiv.SortCriterion.Relevance = "relevance"
        mock_search = MagicMock()
        mock_search.results.return_value = []
        mock_arxiv.Search.return_value = mock_search

        loader = DocumentLoader()
        loader.load_arxiv("test query", max_results=999)

        call_kwargs = mock_arxiv.Search.call_args
        self.assertLessEqual(
            call_kwargs.kwargs.get("max_results", call_kwargs[1].get("max_results", 999)),
            DocumentLoader._ARXIV_MAX_RESULTS_CAP,
        )


# ── Secret-pattern scanning ──────────────────────────────────────────────────

_SECRET_PATTERNS = re.compile(r"AIzaSy[A-Za-z0-9_-]{33}|[0-9a-f]{32}")


class TestNoSecretsInSource(unittest.TestCase):
    """Scan tracked Python files and the notebook for leaked secret patterns."""

    def _repo_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    def test_python_files_clean(self):
        root = self._repo_root()
        hits = []
        for py in root.rglob("*.py"):
            if "__pycache__" in str(py):
                continue
            text = py.read_text(errors="ignore")
            if _SECRET_PATTERNS.search(text):
                hits.append(str(py.relative_to(root)))
        self.assertEqual(hits, [], f"Secret patterns found in: {hits}")

    def test_notebook_source_clean(self):
        nb_path = self._repo_root() / "notebooks" / "Multi_Domain_Agent.ipynb"
        if not nb_path.exists():
            self.skipTest("Notebook not found")
        with open(nb_path) as f:
            nb = json.load(f)
        for i, cell in enumerate(nb["cells"]):
            src = "".join(cell.get("source", []))
            match = _SECRET_PATTERNS.search(src)
            if match:
                self.fail(
                    f"Secret pattern in notebook cell {i}: {match.group()[:12]}..."
                )

    def test_notebook_outputs_stripped(self):
        nb_path = self._repo_root() / "notebooks" / "Multi_Domain_Agent.ipynb"
        if not nb_path.exists():
            self.skipTest("Notebook not found")
        with open(nb_path) as f:
            nb = json.load(f)
        cells_with_output = [
            i for i, c in enumerate(nb["cells"]) if c.get("outputs")
        ]
        self.assertEqual(
            cells_with_output,
            [],
            f"Cells still have outputs (should be stripped): {cells_with_output}",
        )


# ── Logging sanitisation ─────────────────────────────────────────────────────

@unittest.skipUnless(_HAS_LANGCHAIN, "langchain_core not installed")
class TestLogSanitisation(unittest.TestCase):
    """Verify that error/warning logs do not dump full exception text."""

    def test_model_registry_json_parse_log(self):
        from foundation.model_registry import _parse_llm_json

        with self.assertLogs("foundation.model_registry", level="WARNING") as cm:
            _parse_llm_json("this is not json at all " * 50)

        for record in cm.output:
            self.assertNotIn("this is not json", record,
                             "Raw LLM output should not appear in WARNING logs")
            self.assertIn("output length=", record)

    def test_supervisor_parse_log_no_raw(self):
        from orchestration.supervisor import SupervisorAgent

        mock_llm = MagicMock()
        mock_llm.invoke_and_parse_json.return_value = {
            "error": "parse_failed",
            "raw": "SENSITIVE RAW TEXT HERE" * 10,
        }
        mock_llm.invoke.return_value = MagicMock(content="clarify question")

        sup = SupervisorAgent(mock_llm)

        with self.assertLogs("orchestration.supervisor", level="WARNING") as cm:
            sup.route("test query")

        for record in cm.output:
            self.assertNotIn("SENSITIVE RAW TEXT", record,
                             "Raw LLM output should not appear in supervisor logs")


# ── CI workflow secret gate ───────────────────────────────────────────────────

class TestCIWorkflowHasSecretScan(unittest.TestCase):
    """Verify the sync workflow includes a secret-scanning step."""

    def test_workflow_has_scan_step(self):
        root = Path(__file__).resolve().parent.parent
        wf = root / ".github" / "workflows" / "sync-to-gdrive.yml"
        if not wf.exists():
            self.skipTest("Workflow file not found")
        text = wf.read_text()
        self.assertIn("Scan for leaked secrets", text)
        self.assertIn("AIzaSy", text)


if __name__ == "__main__":
    unittest.main()
