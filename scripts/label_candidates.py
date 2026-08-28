"""Interactive relevance labelling for the round-2 candidate pool.

Editing 1,532 ``"relevant": null`` fields by hand invites two mistakes that are
expensive and silent: mis-keying a trial id, and losing your place. This shows
one candidate at a time and writes the file after **every** answer, so a crash
or a Ctrl-C costs nothing and re-running resumes exactly where you stopped.

It does not decide anything. Relevance is the ground truth every metric is
measured against; a machine-generated judgement would be the retrieval system
grading its own homework.

Judge **complete queries** rather than skimming across all of them. Thirty
fully-judged queries make a usable eval set; ninety-seven half-judged ones make
a biased one, because the unjudged remainder silently scores as irrelevant.

Run::

    python scripts/label_candidates.py            # label, resuming
    python scripts/label_candidates.py --status   # progress only
    python scripts/label_candidates.py --merge    # fold into the eval set
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

CANDIDATES = Path("reports/eval_round2_candidates.json")
EVAL_SET = Path("tests/fixtures/eval_queries.json")

HELP = """
  y  relevant        the trial is about this query
  n  not relevant    it is not
  s  skip            undecided; asked again next run
  b  back            undo the previous answer
  q  quit            save and stop
"""


def _safe(text: str) -> str:
    """Render text the Windows console can print without dying on a codec."""
    return text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
        sys.stdout.encoding or "utf-8", errors="replace"
    )


def _load() -> dict[str, Any]:
    if not CANDIDATES.exists():
        raise SystemExit(f"No candidate sheet at {CANDIDATES}. Run scripts/make_eval_round2.py")
    return json.loads(CANDIDATES.read_text(encoding="utf-8"))


def _save(payload: dict[str, Any]) -> None:
    """Write atomically -- a half-written sheet would lose the whole session."""
    temporary = CANDIDATES.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(CANDIDATES)


def _counts(payload: dict[str, Any]) -> tuple[int, int, int]:
    """Return ``(judged, total, fully_judged_queries)``."""
    judged = total = complete = 0
    for query in payload["queries"]:
        done = sum(1 for c in query["candidates"] if c["relevant"] is not None)
        judged += done
        total += len(query["candidates"])
        if done == len(query["candidates"]):
            complete += 1
    return judged, total, complete


def status(payload: dict[str, Any]) -> None:
    judged, total, complete = _counts(payload)
    print(f"\n  judged            {judged} of {total}  ({judged / max(total, 1):.0%})")
    print(f"  complete queries  {complete} of {len(payload['queries'])}")
    print("\n  Only complete queries are safe to merge -- a partially judged query")
    print("  scores its unjudged remainder as irrelevant.\n")


def label(payload: dict[str, Any]) -> None:
    """Walk unjudged candidates, query by query."""
    print(HELP)
    history: list[tuple[int, int]] = []

    for qi, query in enumerate(payload["queries"]):
        pending = [i for i, c in enumerate(query["candidates"]) if c["relevant"] is None]
        if not pending:
            continue

        print("=" * 72)
        print(f"  QUERY {qi + 1}/{len(payload['queries'])}: {_safe(query['query'])}")
        print(
            f"  {query['already_judged_relevant']} already known relevant"
            f" | {len(pending)} left to judge here"
        )
        print("=" * 72)

        index = 0
        while index < len(pending):
            ci = pending[index]
            candidate = query["candidates"][ci]
            print(
                f"\n  [{index + 1}/{len(pending)}]  {candidate['trial_id']}"
                f"   found by: {', '.join(candidate['found_by'])}"
            )
            print(f"  {_safe(candidate['title'])}")
            abstract = candidate.get("abstract") or ""
            if abstract and abstract != candidate["title"]:
                print(f"  {_safe(abstract[:280])}")

            try:
                answer = input("  relevant? [y/n/s/b/q] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  stopped.")
                _save(payload)
                return

            if answer == "q":
                _save(payload)
                print("\n  saved. Re-run to continue where you stopped.")
                return
            if answer == "b":
                if history:
                    pqi, pci = history.pop()
                    payload["queries"][pqi]["candidates"][pci]["relevant"] = None
                    print("  previous answer cleared.")
                    if pqi == qi:
                        index = max(0, index - 1)
                        pending = [
                            i for i, c in enumerate(query["candidates"]) if c["relevant"] is None
                        ]
                continue
            if answer == "s":
                index += 1
                continue
            if answer not in {"y", "n"}:
                print(HELP)
                continue

            candidate["relevant"] = answer == "y"
            history.append((qi, ci))
            _save(payload)  # after every answer: a crash costs nothing
            index += 1

    _save(payload)
    print("\n  every candidate judged.")


def merge(payload: dict[str, Any]) -> None:
    """Fold judged-relevant candidates into the eval set.

    Only **fully judged** queries are merged. Merging a partially judged query
    would add its positives while its unjudged remainder keeps scoring as
    irrelevant -- worse than not merging it at all.
    """
    eval_set = json.loads(EVAL_SET.read_text(encoding="utf-8"))
    # The fixture is {"_provenance": ..., "queries": [...]}, not a bare list.
    entries = eval_set["queries"] if isinstance(eval_set, dict) else eval_set
    by_query = {entry["query"]: entry for entry in entries}

    merged_queries = added = 0
    skipped: list[str] = []

    for query in payload["queries"]:
        unjudged = [c for c in query["candidates"] if c["relevant"] is None]
        if unjudged:
            skipped.append(f"{query['query']} ({len(unjudged)} unjudged)")
            continue
        target = by_query.get(query["query"])
        if target is None:
            continue
        existing = set(target["relevant"])
        new = [
            c["trial_id"]
            for c in query["candidates"]
            if c["relevant"] and c["trial_id"] not in existing
        ]
        if new:
            target["relevant"] = sorted(existing | set(new))
            added += len(new)
        merged_queries += 1

    if not merged_queries:
        print("\n  nothing to merge -- no query is fully judged yet.\n")
        return

    backup = EVAL_SET.with_suffix(".json.bak")
    shutil.copy2(EVAL_SET, backup)
    EVAL_SET.write_text(json.dumps(eval_set, indent=2) + "\n", encoding="utf-8")

    print(f"\n  merged {merged_queries} fully-judged queries, +{added} relevance judgements")
    print(f"  backup: {backup}")
    if skipped:
        print(f"  skipped {len(skipped)} partially-judged queries")
    print("\n  Now re-run: medsearch evaluate\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="Show progress and exit.")
    parser.add_argument("--merge", action="store_true", help="Fold judgements into the eval set.")
    args = parser.parse_args()

    payload = _load()
    if args.status:
        status(payload)
    elif args.merge:
        merge(payload)
    else:
        label(payload)
        status(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
