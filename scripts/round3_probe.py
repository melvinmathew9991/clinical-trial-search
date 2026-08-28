"""Round 3: the two measurements that need no relevance judgements.

Both strata below are scored without a labeller, which matters: every number
in ``reports/evaluation.json`` rests on judgements that are 42 % model-
generated, and EVALUATION_AUDIT.md section 7 says so on every table. These two
do not.

**Negation pairs.** Four queries are paired with their negated twin
("hospitalized" / "non-hospitalized"). If a system returns the same documents
for both, it does not represent negation -- and that is visible as rank
overlap, with no judgement of relevance anywhere in it. A system that models
negation should return *different* documents; one that treats "non-" as noise
returns the same list twice.

**Registry codes.** A document is relevant to the query ``NCT04446429`` iff its
text contains that string. Ground truth is exact, objective and free. All three
codes are cited by other trials -- companion papers, extension studies -- never
by the trial that owns the id, so this is retrieval and not a row lookup.

Run::

    python scripts/round3_probe.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

from medsearch.runtime import configure_threads

configure_threads()

from medsearch.config import get_settings  # noqa: E402
from medsearch.data.loader import load_corpus  # noqa: E402
from medsearch.logging_conf import configure_logging  # noqa: E402
from medsearch.pipelines.train import load_search_engine, run_preprocessing  # noqa: E402
from medsearch.preprocessing.pipeline import TextPreprocessor  # noqa: E402
from medsearch.search.baseline import TfidfBaseline  # noqa: E402
from medsearch.search.bm25 import BM25Baseline  # noqa: E402

FIELD = "abstract"
TOP_N = 10

NEGATION_PAIRS: tuple[tuple[str, str], ...] = (
    ("hospitalized patients with covid-19", "non-hospitalized patients with covid-19"),
    ("patients requiring supplemental oxygen", "patients not requiring supplemental oxygen"),
    ("severe covid-19 pneumonia", "non-severe covid-19 pneumonia"),
    ("treatment with mechanical ventilation", "treatment without mechanical ventilation"),
)

CODES: tuple[str, ...] = ("NCT04446429", "NCT04317092", "NCT04372368")


def _mrr(ranked: list[str], relevant: set[str]) -> float:
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)

    corpus = load_corpus(settings.paths.corpus_file)
    trial_ids = corpus["trial_id"].astype(str).tolist()
    text = (corpus["title"].astype(str) + " " + corpus["abstract"].astype(str)).tolist()
    cache, _, _ = run_preprocessing(settings, cast(Any, FIELD))
    pre = TextPreprocessor()
    tfidf, bm25 = TfidfBaseline(cache), BM25Baseline(cache)
    engines = {
        name: load_search_engine(settings, cast(Any, name), cast(Any, FIELD))
        for name in ("skipgram", "fasttext")
    }

    def rank(system: str, query: str) -> list[str]:
        if system in engines:
            return [r.trial_id for r in engines[system].search(query, top_n=TOP_N).results]
        tokens = pre.transform(query)
        backend = tfidf if system == "tfidf" else bm25
        return [trial_ids[h.row_id] for h in backend.search(tokens, top_n=TOP_N)]

    systems = ("skipgram", "fasttext", "tfidf", "bm25")
    report: dict[str, Any] = {"negation_pairs": [], "registry_codes": []}

    print("\nNEGATION -- overlap@10 between a query and its negated twin")
    print("(1.00 = the negation changed nothing at all)\n")
    print(f"  {'pair':<44}" + "".join(f"{s:>10}" for s in systems))
    for positive, negated in NEGATION_PAIRS:
        row: dict[str, Any] = {"positive": positive, "negated": negated, "overlap": {}}
        line = f"  {negated[:42]:<44}"
        for system in systems:
            a, b = set(rank(system, positive)), set(rank(system, negated))
            overlap = len(a & b) / TOP_N
            row["overlap"][system] = round(overlap, 3)
            line += f"{overlap:>10.2f}"
        report["negation_pairs"].append(row)
        print(line)
    means = {
        s: sum(r["overlap"][s] for r in report["negation_pairs"]) / len(NEGATION_PAIRS)
        for s in systems
    }
    report["negation_mean_overlap"] = {s: round(v, 3) for s, v in means.items()}
    print(f"  {'MEAN':<44}" + "".join(f"{means[s]:>10.2f}" for s in systems))

    print("\n\nREGISTRY CODES -- ground truth is string containment, no judging\n")
    print(f"  {'query':<16}{'|rel|':>6}" + "".join(f"{s:>10}" for s in systems))
    for code in CODES:
        pattern = re.compile(re.escape(code), re.IGNORECASE)
        relevant = {trial_ids[i] for i, t in enumerate(text) if pattern.search(t)}
        row = {"code": code, "n_relevant": len(relevant), "mrr": {}, "recall_at_10": {}}
        line = f"  {code:<16}{len(relevant):>6}"
        for system in systems:
            ranked = rank(system, code)
            hits = len(set(ranked) & relevant)
            row["mrr"][system] = round(_mrr(ranked, relevant), 3)
            row["recall_at_10"][system] = round(hits / max(len(relevant), 1), 3)
            line += f"{row['recall_at_10'][system]:>10.2f}"
        report["registry_codes"].append(row)
        print(line)
    print("  (cells are Recall@10 against the exact-match ground truth)")

    out = Path("reports/round3_probe.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n  wrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
