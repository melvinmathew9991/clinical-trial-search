"""Score known-item retrieval: paste a registry id, get that trial.

Separate from ``round3_evaluate.py`` because it asks a different question of a
different gold set. Round 3's ``code`` stratum scores *citation* finding -- which
trials mention this id -- and its gold lists citing trials only, excluding the
queried trial by construction. This scores the lookup itself.

The ground truth needs no annotator: trial ids are unique across all 10,666
rows, so the answer is defined rather than judged. It is the only stratum in the
project with no provenance to declare.

Metrics come from ``medsearch.pipelines.evaluate`` unchanged, so the numbers are
comparable with every other report. ``report_name=None`` keeps this off
``reports/evaluation.json``.

Run::

    python scripts/known_item_evaluate.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from medsearch.runtime import configure_threads

configure_threads()

from medsearch.config import get_settings  # noqa: E402
from medsearch.logging_conf import configure_logging  # noqa: E402
from medsearch.pipelines.evaluate import run_evaluation  # noqa: E402

FIXTURE = Path("tests/fixtures/eval_queries_known_item.json")
OUTPUT = Path("reports/known_item.json")

#: MRR@10 is the metric to read. With exactly one relevant document per query it
#: equals 1.0 precisely when every query puts that document at rank 1, and
#: unlike Precision@k it is not distorted by the union's doubled scoring depth
#: -- a union row's "P@1" is really P@2 (EVALUATION_AUDIT section 9), so with
#: one relevant document it cannot exceed 0.5 however perfect the ranking.
K_VALUES = (1, 5, 10)


def main() -> None:
    configure_logging()
    settings = get_settings()
    report: dict[str, Any] = run_evaluation(
        settings, eval_path=FIXTURE, k_values=K_VALUES, report_name=None
    )

    queries = report["eval_queries"]
    print(f"\nKNOWN ITEM  --  {queries} registry-id queries, 1 relevant trial each\n")
    print(f"  {'method':<20} {'MRR@10':>8} {'R@10':>7}   rank-1 hit rate")
    for result in report["results"]:
        # One relevant document, so MRR is 1/rank of the answer: 1.000 means
        # every query put it first.
        print(
            f"  {result['method']:<20}"
            f" {result['mrr_at']['10']:>8.3f}"
            f" {result['recall_at']['10']:>7.3f}"
            f"   {result['mrr_at']['10']:>6.1%}"
        )
    print(
        "\n  The lexical baselines are scored without the known-item wrapper: they are"
        "\n  baselines, not the shipped system, and they show what text retrieval alone"
        "\n  can do here -- only 71 of 10,666 abstracts contain any registry code."
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n  wrote {OUTPUT}\n")


if __name__ == "__main__":
    main()
