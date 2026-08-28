"""Tests for the function-length gate.

A gate nobody tests is a gate that can silently stop failing. This one exists
precisely because Rules.md section 3 went eleven sprints unenforced, so the
counting rule it implements is pinned here rather than trusted.

The script lives in ``scripts/`` and is not importable as a package, so it is
loaded by path -- the same way the pre-commit hook and CI invoke it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_function_length.py"


def _load() -> ModuleType:
    """Import the checker from its path."""
    spec = importlib.util.spec_from_file_location("check_function_length", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load()


def _measure(source: str, tmp_path: Path) -> int:
    """Run the counter over a single function defined in ``source``."""
    import ast

    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    function = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
    return int(checker.body_length(function, lines))


class TestCountingRule:
    """What counts toward the cap, and what deliberately does not."""

    def test_counts_executable_lines(self, tmp_path: Path) -> None:
        assert _measure("def f():\n    a = 1\n    b = 2\n    return a + b\n", tmp_path) == 3

    def test_docstring_is_excluded(self, tmp_path: Path) -> None:
        """A well-documented function must not be penalised for it."""
        source = 'def f():\n    """One.\n\n    Two.\n    Three.\n    """\n    return 1\n'
        assert _measure(source, tmp_path) == 1

    def test_blank_and_comment_lines_are_excluded(self, tmp_path: Path) -> None:
        source = "def f():\n    a = 1\n\n    # explanation\n    # continued\n    return a\n"
        assert _measure(source, tmp_path) == 2

    def test_a_docstring_only_function_measures_zero(self, tmp_path: Path) -> None:
        assert _measure('def f():\n    """Nothing here."""\n', tmp_path) == 0

    def test_multiline_statement_counts_every_line(self, tmp_path: Path) -> None:
        """A wrapped call is as much to read as the lines it occupies."""
        source = "def f():\n    return dict(\n        a=1,\n        b=2,\n    )\n"
        assert _measure(source, tmp_path) == 4


class TestReporting:
    """Severity, exit codes, and which functions are surfaced at all."""

    def _write(self, tmp_path: Path, body_lines: int) -> Path:
        body = "\n".join(f"    x{i} = {i}" for i in range(body_lines))
        (tmp_path / "m.py").write_text(f"def big():\n{body}\n", encoding="utf-8")
        return tmp_path

    def test_functions_within_the_soft_cap_are_not_reported(self, tmp_path: Path) -> None:
        assert checker.collect([str(self._write(tmp_path, checker.SOFT_CAP))]) == []

    def test_over_the_soft_cap_warns_without_failing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        found = checker.collect([str(self._write(tmp_path, checker.SOFT_CAP + 1))])
        assert len(found) == 1
        assert not found[0].over_hard_cap

    def test_over_the_hard_cap_is_a_failure(self, tmp_path: Path) -> None:
        found = checker.collect([str(self._write(tmp_path, checker.HARD_CAP + 1))])
        assert len(found) == 1
        assert found[0].over_hard_cap

    def test_exactly_at_the_hard_cap_still_passes(self, tmp_path: Path) -> None:
        """The cap is inclusive -- 60 lines is allowed, 61 is not."""
        found = checker.collect([str(self._write(tmp_path, checker.HARD_CAP))])
        assert [o.over_hard_cap for o in found] == [False]


def test_the_project_itself_is_within_the_hard_cap() -> None:
    """The gate's own subject. Fails loudly rather than drifting again."""
    root = SCRIPT.resolve().parents[1] / "src" / "medsearch"
    over = [o for o in checker.collect([str(root)]) if o.over_hard_cap]
    assert over == [], f"over the {checker.HARD_CAP}-line hard cap: {over}"
