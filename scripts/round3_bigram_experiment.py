"""Does adding bigram features fix free-standing negation?

The hypothesis comes straight out of EVALUATION_AUDIT.md section 8. Prefix
negation works because hyphen-joining mints a *rare* token: ``nonhospitalized``
carries idf 6.45 against ``hospitalized``'s 3.14, so it dominates the query
vector. Free-standing negation fails because ``not`` is retained but carries
idf 2.18 -- less information than the words it is supposed to invert.

As a bigram, the same negation becomes rare again:

===================  ======  =====
feature              df      idf
===================  ======  =====
``not`` (unigram)     3,274   2.18
``not requiring``        71   6.00
``not hospitalized``     26   6.98
``without mechanical``    9   7.97
``nonhospitalized``      45   6.45
===================  ======  =====

So the prediction is specific: bigrams should move the two negation pairs that
free-standing negation governs, and should leave the prefix-negation pairs
roughly where they already are. A change that improves everything equally would
be evidence the mechanism is *not* what section 8 claims.

This is a feature-engineering change only. The rankers are untouched: both
baselines take token lists, so a bigram condition is built by augmenting the
token lists and rebuilding. Metrics come from ``evaluate_baseline`` unchanged,
so they are comparable with ``reports/evaluation.json`` and with round 3.

**Regression matters as much as the effect.** Bigrams triple the vocabulary,
so this also scores the entity and code strata and the full 97-query set, to
see what the change costs where it was not aimed.

Run::

    python scripts/round3_bigram_experiment.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from medsearch.runtime import configure_threads

configure_threads()

from medsearch.config import get_settings  # noqa: E402
from medsearch.data.loader import load_corpus  # noqa: E402
from medsearch.logging_conf import configure_logging  # noqa: E402
from medsearch.pipelines.evaluate import evaluate_baseline, load_eval_set  # noqa: E402
from medsearch.pipelines.train import run_preprocessing  # noqa: E402
from medsearch.preprocessing.pipeline import TextPreprocessor  # noqa: E402
from medsearch.search.baseline import TfidfBaseline  # noqa: E402
from medsearch.search.bm25 import BM25Baseline  # noqa: E402

FIELD = "abstract"
TOP_N = 10
ROUND3 = Path("tests/fixtures/eval_queries_round3.json")
MAIN = Path("tests/fixtures/eval_queries.json")

NEGATION_PAIRS = (
    ("hospitalized patients with covid-19", "non-hospitalized patients with covid-19", "prefix"),
    (
        "patients requiring supplemental oxygen",
        "patients not requiring supplemental oxygen",
        "free-standing",
    ),
    ("severe covid-19 pneumonia", "non-severe covid-19 pneumonia", "prefix"),
    (
        "treatment with mechanical ventilation",
        "treatment without mechanical ventilation",
        "free-standing",
    ),
)


def with_bigrams(tokens: list[str]) -> list[str]:
    """Unigrams plus adjacent bigrams, joined by a space.

    The vocabulary is a ``dict[str, int]``, so a bigram is just a longer key --
    no change to either ranker is needed to carry it.
    """
    return tokens + [f"{a} {b}" for a, b in pairwise(tokens)]


class BigramPreprocessor:
    """Wraps the real preprocessor so queries get the same features as documents."""

    def __init__(self, inner: TextPreprocessor) -> None:
        self._inner = inner

    def transform(self, text: str) -> list[str]:
        return with_bigrams(self._inner.transform(text))


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)

    corpus = load_corpus(settings.paths.corpus_file)
    trial_ids = corpus["trial_id"].astype(str).tolist()
    cache, _, _ = run_preprocessing(settings, cast(Any, FIELD))
    unigram_docs = [list(d) for d in cache]
    bigram_docs = [with_bigrams(d) for d in unigram_docs]

    plain = TextPreprocessor()
    conditions: dict[str, dict[str, Any]] = {
        "unigram": {"pre": plain, "docs": unigram_docs},
        "bigram": {"pre": BigramPreprocessor(plain), "docs": bigram_docs},
    }
    for name, block in conditions.items():
        docs = block["docs"]
        block["tfidf"] = TfidfBaseline(docs)
        block["bm25"] = BM25Baseline(docs)
        block["vocab"] = len({t for doc in docs for t in doc})
        print(f"  {name:8s} vocabulary {block['vocab']:>7,}")

    report: dict[str, Any] = {
        "vocabulary": {k: v["vocab"] for k, v in conditions.items()},
        "negation_pairs": [],
        "strata": {},
        "main_set": {},
    }

    def rank(condition: str, system: str, query: str) -> list[str]:
        block = conditions[condition]
        tokens = block["pre"].transform(query)
        return [trial_ids[h.row_id] for h in block[system].search(tokens, top_n=TOP_N)]

    print("\n\nNEGATION PAIRS -- overlap@10 with the negated twin (lower is better)\n")
    print(
        f"  {'pair':<40}{'kind':<15}{'tfidf uni':>11}{'tfidf bi':>10}{'bm25 uni':>10}{'bm25 bi':>9}"
    )
    for positive, negated, kind in NEGATION_PAIRS:
        row: dict[str, Any] = {"negated": negated, "kind": kind, "overlap": {}}
        line = f"  {negated[:38]:<40}{kind:<15}"
        for system in ("tfidf", "bm25"):
            for condition in ("unigram", "bigram"):
                value = (
                    len(
                        set(rank(condition, system, positive))
                        & set(rank(condition, system, negated))
                    )
                    / TOP_N
                )
                row["overlap"][f"{system}-{condition}"] = round(value, 3)
                line += f"{value:>11.2f}" if system == "tfidf" else f"{value:>10.2f}"
        report["negation_pairs"].append(row)
        print(line)

    def score(queries: Any, label: str) -> list[dict[str, Any]]:
        rows = []
        for system in ("tfidf", "bm25"):
            for condition in ("unigram", "bigram"):
                block = conditions[condition]
                result = evaluate_baseline(
                    block[system],
                    queries,
                    block["pre"],
                    trial_ids,
                    name=f"{system}-{condition}",
                    k_values=(1, 5, 10),
                )
                rows.append(asdict(result))
        print(f"\n\n{label}\n")
        print(
            f"  {'system':<18}{'P@1':>8}{'MRR@10':>9}{'nDCG@10':>9}{'R@10':>8}{'R-prec':>9}{'p95 ms':>9}"
        )
        for row in rows:
            print(
                f"  {row['method']:<18}"
                f"{row['precision_at'][1]:>8.3f}{row['mrr_at'][10]:>9.3f}"
                f"{row['ndcg_at'][10]:>9.3f}{row['recall_at'][10]:>8.3f}"
                f"{row['r_precision']:>9.3f}{row['latency_ms']['p95']:>9.1f}"
            )
        return rows

    round3 = json.loads(ROUND3.read_text(encoding="utf-8"))["queries"]
    for stratum in ("negation", "entity", "code"):
        subset = [q for q in round3 if q["stratum"] == stratum]
        tmp = Path(f"reports/_tmp_{stratum}.json")
        tmp.write_text(json.dumps({"queries": subset}), encoding="utf-8")
        try:
            report["strata"][stratum] = score(
                load_eval_set(tmp), f"ROUND-3 {stratum.upper()} STRATUM ({len(subset)} queries)"
            )
        finally:
            tmp.unlink()

    report["main_set"] = score(load_eval_set(MAIN), "MAIN SET (97 queries) -- regression check")

    out = Path("reports/round3_bigram_experiment.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n  wrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
