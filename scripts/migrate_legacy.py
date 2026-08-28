"""One-shot migration from the legacy Part_1 / Part_2 layout.

Moves the source corpus into ``data/raw/`` and reports the legacy artefacts
that are deliberately *not* migrated (Architecture.md section 12).

Nothing is discarded by this script. Artefacts that v1 regenerates are listed
with their sizes so the operator can delete them once the new pipeline has
been verified -- that call belongs to a human, not to a migration script.

Usage::

    python scripts/migrate_legacy.py --dry-run     # default, shows the plan
    python scripts/migrate_legacy.py --apply
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_ROOT = REPO_ROOT.parent

#: (legacy relative path, destination relative to repo root)
MOVE_PLAN: tuple[tuple[str, str], ...] = (
    (
        "Part_1/Data/Data/Dimension-covid.csv",
        "data/raw/dimension-covid.csv",
    ),
    (
        "Part_1/Ipython Notebook/Ipython Notebook/Medical Embeddings_Final.ipynb",
        "notebooks/01-exploration.ipynb",
    ),
)

#: Legacy artefacts v1 regenerates. Listed, never auto-deleted.
SUPERSEDED: tuple[tuple[str, str], ...] = (
    (
        "Part_1/Modular+Code/Modular Code/Medical_Embeddings/output",
        "Models and CSV vectors -- retrained under ADR-001 (bounded bucket) and "
        "ADR-002 (.npy index). Includes the 800 MB vectors_ngrams.npy.",
    ),
    (
        "Part_1/Data/Data/skipgram-vec.csv",
        "Document vectors -- rebuilt as float32 .npy, ~5x smaller.",
    ),
    (
        "Part_1/Data/Data/FastText-vec.csv",
        "Document vectors -- rebuilt as float32 .npy.",
    ),
)


@dataclass(frozen=True, slots=True)
class Action:
    source: Path
    destination: Path
    size_mb: float
    exists: bool


def _size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1024**2
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**2
    return 0.0


def plan_moves() -> list[Action]:
    actions: list[Action] = []
    for rel_source, rel_dest in MOVE_PLAN:
        source = LEGACY_ROOT / rel_source
        destination = REPO_ROOT / rel_dest
        actions.append(
            Action(
                source=source,
                destination=destination,
                size_mb=_size_mb(source),
                exists=source.exists(),
            )
        )
    return actions


def report_superseded() -> list[tuple[Path, float, str]]:
    rows: list[tuple[Path, float, str]] = []
    for rel_path, reason in SUPERSEDED:
        path = LEGACY_ROOT / rel_path
        if path.exists():
            rows.append((path, _size_mb(path), reason))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dry-run", action="store_true", default=True)
    group.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy instead of move, leaving the legacy tree untouched (needs ~30 MB extra).",
    )
    args = parser.parse_args(argv)
    apply = args.apply

    print(f"Legacy root : {LEGACY_ROOT}")
    print(f"Repo root   : {REPO_ROOT}")
    print(f"Mode        : {'APPLY' if apply else 'DRY RUN'}"
          f"{' (copy)' if args.copy else ' (move)'}\n")

    actions = plan_moves()
    print("Files to migrate")
    print("-" * 72)
    missing = 0
    for action in actions:
        if not action.exists:
            print(f"  MISSING  {action.source}")
            missing += 1
            continue
        verb = "copy" if args.copy else "move"
        print(f"  {verb:<5} {action.size_mb:>8.1f} MB  {action.source.name}")
        print(f"        ->  {action.destination.relative_to(REPO_ROOT)}")
        if apply:
            action.destination.parent.mkdir(parents=True, exist_ok=True)
            if action.destination.exists():
                print("        SKIPPED - destination already exists")
                continue
            if args.copy:
                shutil.copy2(action.source, action.destination)
            else:
                shutil.move(str(action.source), str(action.destination))
            print("        done")

    superseded = report_superseded()
    if superseded:
        total = sum(size for _, size, _ in superseded)
        print(f"\nLegacy artefacts NOT migrated ({total:.0f} MB reclaimable)")
        print("-" * 72)
        for path, size, reason in superseded:
            print(f"  {size:>8.1f} MB  {path.relative_to(LEGACY_ROOT)}")
            print(f"              {reason}")
        print(
            "\n  These are left in place. Delete them yourself once "
            "`make train` has produced working artefacts."
        )

    if not apply:
        print("\nDry run only. Re-run with --apply to perform the migration.")
    elif missing == 0:
        print("\nMigration complete. Next: `medsearch doctor`, then `make train`.")

    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
