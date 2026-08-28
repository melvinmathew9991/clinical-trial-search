"""Score the round-3 eval set, one stratum at a time.

The three strata are different retrieval tasks. ``entity`` is ad-hoc search with
~13 relevant documents per query; ``code`` is known-item search with 2; and
``negation`` is a discrimination task where the interesting quantity is how a
query differs from its twin. Averaging them would produce a number that
describes none of them, which is the mistake EVALUATION_AUDIT.md section 7 was
written about -- so this reports each separately and never sums them.

Metrics come from ``medsearch.pipelines.evaluate`` unchanged: this script slices
the fixture and delegates, so the numbers are produced by the same code that
writes ``reports/evaluation.json`` and are comparable with it.

Run::

    python scripts/round3_evaluate.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from medsearch.runtime import configure_threads

configure_threads()

from medsearch.config import get_settings  # noqa: E402
from medsearch.logging_conf import configure_logging  # noqa: E402
from medsearch.pipelines.evaluate import run_evaluation  # noqa: E402

FIXTURE = Path("tests/fixtures/eval_queries_round3.json")
STRATA = ("entity", "code", "negation")
#: Known-item search over 2 relevant documents cannot use k=10 sensibly; the
#: metric that matters there is whether the right document is at rank 1.
K_VALUES = (1, 5, 10)


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    queries = payload["queries"]

    report: dict[str, Any] = {"fixture": str(FIXTURE), "strata": {}}

    with tempfile.TemporaryDirectory() as tmp:
        for stratum in STRATA:
            subset = [q for q in queries if q["stratum"] == stratum]
            path = Path(tmp) / f"{stratum}.json"
            path.write_text(json.dumps({"queries": subset}), encoding="utf-8")
            result = run_evaluation(settings, eval_path=path, k_values=K_VALUES)
            report["strata"][stratum] = {
                "queries": len(subset),
                "mean_relevant": round(sum(len(q["relevant"]) for q in subset) / len(subset), 1),
                "results": result["results"],
            }

    for stratum in STRATA:
        block = report["strata"][stratum]
        print(
            f"\n\n{stratum.upper()}  --  {block['queries']} queries, "
            f"{block['mean_relevant']} relevant per query on average\n"
        )
        print(f"  {'method':<18}{'P@1':>8}{'MRR@10':>9}{'nDCG@10':>9}{'R@10':>8}{'R-prec':>9}")
        for row in block["results"]:
            print(
                f"  {row['method']:<18}"
                f"{row['precision_at']['1']:>8.3f}"
                f"{row['mrr_at']['10']:>9.3f}"
                f"{row['ndcg_at']['10']:>9.3f}"
                f"{row['recall_at']['10']:>8.3f}"
                f"{row['r_precision']:>9.3f}"
            )

    out = Path("reports/evaluation_round3.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n  wrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
