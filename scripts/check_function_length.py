"""Enforce the Rules.md section 3 function-length caps.

Rules.md has said "soft cap 40 lines; hard cap 60" since Sprint 0. Nothing
enforced it, so by the end of Sprint 8 two functions were over the hard cap
(``doctor`` at 69, ``check_artefacts`` at 61) and eight more were over the
soft cap. A rule no tool checks is a preference.

A first pass at this audit reported *nine* over the hard cap. That count
charged blank and comment lines against the cap, which in a codebase this
heavily commented inflates every function -- the reason the counting rule
below is stated explicitly rather than left to intuition.

Ruff cannot express this: its nearest rule, ``PLR0915``, counts statements
rather than lines, so a single 30-line dict literal reads as one statement
while five short guard clauses read as five. Counting lines is the rule that
was actually written, so this counts lines.

**What is counted.** Body lines only -- the docstring is excluded, along with
blank lines and comment-only lines. This project documents heavily and states
units and shapes in every docstring (Rules section 3); charging that against a
length cap would push authors to write worse docstrings to satisfy a rule
about function complexity. Decorators and the signature are excluded for the
same reason: a Typer command's twelve ``typer.Option`` parameters are an
interface, not a body.

Exit codes: 0 clean or soft-cap warnings only, 1 if anything exceeds the hard
cap. Run: ``python scripts/check_function_length.py [paths...]``
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

SOFT_CAP = 40
HARD_CAP = 60

DEFAULT_PATHS = ("src/medsearch",)


@dataclass(frozen=True, slots=True)
class Offender:
    """One function that exceeds a cap."""

    path: Path
    line: int
    name: str
    length: int

    @property
    def over_hard_cap(self) -> bool:
        """True when this breaks the build rather than merely warning."""
        return self.length > HARD_CAP


def body_length(node: ast.FunctionDef | ast.AsyncFunctionDef, source: list[str]) -> int:
    """Count executable lines in a function body.

    Args:
        node: The function to measure.
        source: The file's lines, 0-indexed.

    Returns:
        Lines in the body, excluding the docstring, blanks, and comment-only
        lines. Nested function definitions count toward their parent, which is
        deliberate -- a closure does not make the enclosing function shorter to
        read.
    """
    body = node.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if not body:
        return 0

    start = body[0].lineno
    end = max(getattr(stmt, "end_lineno", stmt.lineno) or stmt.lineno for stmt in body)

    counted = 0
    for raw in source[start - 1 : end]:
        stripped = raw.strip()
        if stripped and not stripped.startswith("#"):
            counted += 1
    return counted


def scan(path: Path) -> list[Offender]:
    """Measure every function in one file."""
    source = path.read_text(encoding="utf-8").splitlines()
    tree = ast.parse("\n".join(source))

    found: list[Offender] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        length = body_length(node, source)
        if length > SOFT_CAP:
            found.append(Offender(path, node.lineno, node.name, length))
    return found


def collect(paths: list[str]) -> list[Offender]:
    """Measure every function under the given paths."""
    offenders: list[Offender] = []
    for entry in paths:
        root = Path(entry)
        files = sorted(root.rglob("*.py")) if root.is_dir() else [root]
        for file in files:
            offenders.extend(scan(file))
    return sorted(offenders, key=lambda o: -o.length)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", default=list(DEFAULT_PATHS))
    args = parser.parse_args()

    offenders = collect(args.paths or list(DEFAULT_PATHS))
    failures = [o for o in offenders if o.over_hard_cap]
    warnings = [o for o in offenders if not o.over_hard_cap]

    for offender in failures:
        print(
            f"{offender.path}:{offender.line}: {offender.name} is "
            f"{offender.length} lines (hard cap {HARD_CAP})"
        )
    for offender in warnings:
        print(
            f"{offender.path}:{offender.line}: {offender.name} is "
            f"{offender.length} lines (soft cap {SOFT_CAP}) [warning]"
        )

    if failures:
        print(f"\n{len(failures)} function(s) over the {HARD_CAP}-line hard cap.")
        return 1
    if warnings:
        print(f"\n{len(warnings)} over the soft cap, none over the hard cap.")
    else:
        print(f"All functions within the {SOFT_CAP}-line soft cap.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
