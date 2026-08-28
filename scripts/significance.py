"""Paired bootstrap test between two retrieval methods.

Sprint 8 has now needed this three times -- Skip-gram vs FastText, mean vs SIF,
TF-IDF vs the union -- and the first two times it was written ad hoc and thrown
away, which is why the numbers in PRD §8 cannot be re-derived from the repo.
This makes it reproducible.

**Why paired and not a t-test.** The two methods are scored on the *same*
queries, so the per-query difference removes query difficulty from the
comparison. Recall@10 is also bounded in [0, 1] and heavily zero-inflated here,
which is exactly where a normality assumption misleads. Resampling the paired
differences assumes nothing about their shape.

The p-value is two-sided, computed by centring the bootstrap distribution on
zero and asking how often it exceeds the observed effect -- the standard
percentile construction, not a normal approximation.

Run::

    python scripts/significance.py union-skipgram union-fasttext
    python scripts/significance.py tfidf-baseline union-fasttext --metric recall

Method names are ``skipgram``, ``fasttext``, ``union-skipgram``,
``union-fasttext``, ``tfidf-baseline``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np

from medsearch.config import get_settings
from medsearch.logging_conf import configure_logging, get_logger
from medsearch.pipelines.evaluate import EvalQuery, load_eval_set

logger = get_logger(__name__)

FIELD = "abstract"
K = 10
RESAMPLES = 20_000
SEED = 42


def _ranked_ids(name: str, queries: list[EvalQuery], depth: int) -> list[list[str]]:
    """Return each query's ranked trial ids under one method.

    The TF-IDF baseline is the odd one out: it ranks *preprocessed tokens*
    against row indices rather than answering a raw query with trial ids, so it
    needs the corpus id map and the same ``TextPreprocessor`` the corpus was
    built with. Feeding it a differently tokenised query would measure the
    tokeniser instead of the method.
    """
    from medsearch.data.loader import load_corpus
    from medsearch.pipelines.train import (
        load_search_engine,
        load_union_retriever,
        run_preprocessing,
    )

    settings = get_settings()

    if name == "tfidf-baseline":
        from medsearch.preprocessing.pipeline import TextPreprocessor
        from medsearch.search.baseline import TfidfBaseline

        corpus = load_corpus(settings.paths.corpus_file)
        trial_ids = corpus["trial_id"].astype(str).tolist()
        cache, _, _ = run_preprocessing(settings, cast(Any, FIELD))
        baseline = TfidfBaseline(cache)
        preprocessor = TextPreprocessor()
        return [
            [
                trial_ids[h.row_id]
                for h in baseline.search(preprocessor.transform(q.query), top_n=depth)
            ]
            for q in queries
        ]

    if name.startswith("union-"):
        union = load_union_retriever(settings, cast(Any, name[6:]), cast(Any, FIELD))
        return [
            [r.trial_id for r in union.search(q.query, per_method=K).results][:depth]
            for q in queries
        ]

    engine = load_search_engine(settings, cast(Any, name), cast(Any, FIELD))
    return [[r.trial_id for r in engine.search(q.query, top_n=depth).results] for q in queries]


def _per_query(name: str, queries: list[EvalQuery], metric: str) -> np.ndarray:
    """Score one method query by query.

    The union is scored to depth ``2K`` for the same reason the evaluation
    pipeline does it: the union of two top-K lists *is* a 2K-document budget,
    and scoring it to K would measure the truncation rather than the method.
    """
    depth = K * 2 if name.startswith("union-") else K
    rankings = _ranked_ids(name, queries, depth)

    scores: list[float] = []
    for item, ranked in zip(queries, rankings, strict=True):
        relevant = set(item.relevant)
        if metric == "recall":
            hit = len(relevant.intersection(ranked)) / len(relevant) if relevant else 0.0
        else:  # reciprocal rank
            hit = next((1.0 / i for i, t in enumerate(ranked, 1) if t in relevant), 0.0)
        scores.append(hit)
    return np.asarray(scores, dtype=float)


def _paired_bootstrap(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """Return ``(effect, ci_low, ci_high, p)`` for ``b - a``."""
    diff = b - a
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, diff.size, size=(RESAMPLES, diff.size))
    means = diff[idx].mean(axis=1)
    effect = float(diff.mean())
    low, high = (float(x) for x in np.percentile(means, [2.5, 97.5]))
    # Two-sided p: centre the distribution on the null and count the tail.
    centred = means - effect
    p = float((np.abs(centred) >= abs(effect)).mean())
    return effect, low, high, p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--metric", choices=("recall", "mrr"), default="recall")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    queries = load_eval_set(Path("tests/fixtures/eval_queries.json"))

    a = _per_query(args.baseline, queries, args.metric)
    b = _per_query(args.candidate, queries, args.metric)
    effect, low, high, p = _paired_bootstrap(a, b)

    label = f"{args.metric}@{K}"
    print(f"\n{len(queries)} paired queries, {RESAMPLES:,} resamples\n")
    print(f"  {args.baseline:<18} {label} {a.mean():.4f}")
    print(f"  {args.candidate:<18} {label} {b.mean():.4f}")
    print(f"\n  effect {effect:+.4f}   95% CI [{low:+.4f}, {high:+.4f}]   p = {p:.4f}")
    verdict = "significant" if p < 0.05 else "not significant"
    print(f"  -> {verdict} at alpha = 0.05\n")


if __name__ == "__main__":
    main()
