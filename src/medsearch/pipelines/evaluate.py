"""Retrieval evaluation: Recall@k, MRR@k, latency, OOV rate.

Answers the question the project exists to answer -- *does an in-domain
embedding beat keyword search on clinical text?* -- with numbers rather than
eyeballed examples.

**Nothing here invents relevance judgements.** The eval set is supplied by a
human (see ``scripts/make_eval_candidates.py`` and Architecture.md §3.1). If
it is missing, this module says so and stops; it never falls back to a
heuristic that would quietly favour one method over the other.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medsearch.config import FieldName, ModelName, Pooling, Settings
from medsearch.exceptions import DataError
from medsearch.logging_conf import get_logger, stage

logger = get_logger(__name__)

#: Cut-offs reported for every ranked metric.
DEFAULT_K_VALUES: tuple[int, ...] = (1, 5, 10)


@dataclass(frozen=True, slots=True)
class EvalQuery:
    """One labelled query and the trial ids judged relevant to it.

    Attributes:
        query: Free-text query, as a user would type it.
        relevant: Trial ids a human judged relevant. Order is irrelevant.
        note: Optional rationale from the labeller, carried through for audit.
    """

    query: str
    relevant: frozenset[str]
    note: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalQuery:
        """Build from one entry of ``eval_queries.json``."""
        if "query" not in payload or "relevant" not in payload:
            msg = f"Eval entry needs 'query' and 'relevant' keys, got {sorted(payload)}"
            raise DataError(msg)
        return cls(
            query=str(payload["query"]),
            relevant=frozenset(str(r) for r in payload["relevant"]),
            note=str(payload.get("note", "")),
        )


@dataclass(frozen=True, slots=True)
class MethodResult:
    """Metrics for one retrieval method over the whole eval set."""

    method: str
    queries: int
    recall_at: dict[int, float]
    mrr_at: dict[int, float]
    precision_at: dict[int, float]
    #: nDCG@k. The metric the IR literature would use here and the one this
    #: harness lacked: unlike Recall@k it is not capped by |relevant| > k, and
    #: unlike MRR it rewards every relevant document rather than only the
    #: first. Binary gains, log2 discount, ideal ranking as the denominator.
    ndcg_at: dict[int, float]
    #: R-precision: precision at |relevant|, per query. Self-normalising --
    #: a query with 19 relevant documents is scored at depth 19, a query with
    #: 3 at depth 3 -- so it sidesteps the ceiling problem entirely.
    r_precision: float
    #: Mean of min(k, |relevant|) / |relevant|: the best Recall@k any system
    #: could score on this eval set. 44 of 97 queries hold more than 10
    #: relevant documents, so Recall@10 cannot reach 1.0 and must be read
    #: against this number rather than against perfection.
    recall_ceiling_at: dict[int, float]
    latency_ms: dict[str, float]
    unanswered: int
    #: Mean documents actually returned per query. A union retriever asked for
    #: k per method returns up to 2k, so this is what makes its recall legible.
    results_shown: float = 0.0
    #: Documents examined per unit of k. 1 for a single ranker; 2 for the union
    #: of two top-k lists, whose budget is 2k by construction.
    depth_factor: int = 1

    def as_json(self) -> dict[str, Any]:
        """JSON-serialisable form with string keys."""
        payload = asdict(self)
        for key in ("recall_at", "mrr_at", "precision_at", "ndcg_at", "recall_ceiling_at"):
            payload[key] = {str(k): round(v, 4) for k, v in payload[key].items()}
        payload["latency_ms"] = {k: round(v, 3) for k, v in payload["latency_ms"].items()}
        return payload

    def summary_line(self) -> str:
        """One-line rendering for the CLI."""
        return (
            f"  {self.method:<22} "
            f"R@10 {self.recall_at.get(10, 0.0):.3f}  "
            f"MRR@10 {self.mrr_at.get(10, 0.0):.3f}  "
            f"nDCG@10 {self.ndcg_at.get(10, 0.0):.3f}  "
            f"p95 {self.latency_ms.get('p95', 0.0):>7.2f} ms  "
            f"docs {self.results_shown:>4.1f}"
        )


@dataclass
class _Accumulator:
    """Running totals for one method while the eval set is streamed."""

    hits_at: dict[int, list[float]] = field(default_factory=dict)
    reciprocal_at: dict[int, list[float]] = field(default_factory=dict)
    precision_at: dict[int, list[float]] = field(default_factory=dict)
    latencies: list[float] = field(default_factory=list)
    returned: list[int] = field(default_factory=list)
    ndcg_at: dict[int, list[float]] = field(default_factory=dict)
    ceiling_at: dict[int, list[float]] = field(default_factory=dict)
    r_precision: list[float] = field(default_factory=list)
    unanswered: int = 0


def load_eval_set(path: Path) -> list[EvalQuery]:
    """Load and validate the labelled eval set.

    Args:
        path: ``eval_queries.json``.

    Returns:
        Parsed queries, each with at least one relevant id.

    Raises:
        DataError: The file is missing, malformed, or contains a query with no
            relevant documents. An unlabelled entry silently scoring 0 would
            drag every metric down and look like a retrieval failure.
    """
    if not path.exists():
        raise DataError(
            f"No evaluation set at {path}.\n"
            f"  Evaluation needs human relevance judgements; they are not\n"
            f"  generated automatically (see Architecture.md section 3.1).\n"
            f"  Fix: run `python scripts/make_eval_candidates.py` to produce a\n"
            f"  candidate sheet, label it, and save it as {path.name}."
        )

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Evaluation set at {path} is not valid JSON: {exc}"
        raise DataError(msg) from exc

    entries = payload["queries"] if isinstance(payload, dict) else payload
    if not entries:
        msg = f"Evaluation set at {path} contains no queries."
        raise DataError(msg)

    queries = [EvalQuery.from_dict(entry) for entry in entries]

    unlabelled = [q.query for q in queries if not q.relevant]
    if unlabelled:
        raise DataError(
            f"{len(unlabelled)} eval queries have no relevant documents: "
            f"{unlabelled[:3]}...\n"
            f"  An unlabelled query scores 0 on every metric and would read as a\n"
            f"  retrieval failure. Label them or remove them."
        )

    logger.info("Loaded %d labelled eval queries from %s", len(queries), path.name)
    return queries


def _update(
    accumulator: _Accumulator,
    retrieved: Sequence[str],
    relevant: frozenset[str],
    k_values: Sequence[int],
    depth_factor: int = 1,
) -> None:
    """Fold one query's ranked ids into the running totals.

    Args:
        depth_factor: Documents examined per unit of k. The union of two top-k
            lists holds up to 2k documents, so it is scored to depth ``2k`` and
            filed under the label ``k`` -- k is the knob the caller sets, and
            the wider budget is the mechanism being measured, not a leak. Every
            method's metrics therefore stay comparable at a given k, provided
            ``results_shown`` is read alongside them.
    """
    accumulator.returned.append(len(retrieved))
    for k in k_values:
        top_k = retrieved[: k * depth_factor]
        found = [doc for doc in top_k if doc in relevant]

        # Recall@k: fraction of relevant documents surfaced within the budget.
        accumulator.hits_at.setdefault(k, []).append(len(found) / len(relevant))

        # Precision@k: fraction of what was *shown* that is relevant. Dividing
        # by k rather than by the slice length would penalise a short result
        # list twice -- once in recall, again here.
        accumulator.precision_at.setdefault(k, []).append(len(found) / len(top_k) if top_k else 0.0)

        # Reciprocal rank: 1/rank of the first relevant hit, else 0.
        reciprocal = 0.0
        for position, doc in enumerate(top_k, start=1):
            if doc in relevant:
                reciprocal = 1.0 / position
                break
        accumulator.reciprocal_at.setdefault(k, []).append(reciprocal)

        # nDCG@k with binary gains. The ideal ranking puts every relevant
        # document first, so the denominator is the discounted sum over
        # min(|relevant|, budget) positions -- which is what stops a query
        # with more relevant documents than slots from being scored as a
        # failure the way Recall@k does.
        budget = k * depth_factor
        dcg = sum(1.0 / math.log2(i + 1) for i, doc in enumerate(top_k, 1) if doc in relevant)
        ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), budget) + 1))
        accumulator.ndcg_at.setdefault(k, []).append(dcg / ideal if ideal else 0.0)

        # The best Recall@k anyone could score on this query.
        accumulator.ceiling_at.setdefault(k, []).append(min(budget, len(relevant)) / len(relevant))

    # R-precision: precision at depth |relevant|. Independent of k, so it is
    # computed once rather than per k.
    cut = retrieved[: len(relevant)]
    accumulator.r_precision.append(
        sum(1 for doc in cut if doc in relevant) / len(relevant) if relevant else 0.0
    )


def _finalise(
    name: str, accumulator: _Accumulator, count: int, depth_factor: int = 1
) -> MethodResult:
    """Turn running totals into a :class:`MethodResult`."""
    latencies = sorted(accumulator.latencies)

    def percentile(fraction: float) -> float:
        if not latencies:
            return 0.0
        return latencies[min(int(len(latencies) * fraction), len(latencies) - 1)]

    return MethodResult(
        method=name,
        queries=count,
        recall_at={k: statistics.mean(v) for k, v in sorted(accumulator.hits_at.items())},
        mrr_at={k: statistics.mean(v) for k, v in sorted(accumulator.reciprocal_at.items())},
        precision_at={k: statistics.mean(v) for k, v in sorted(accumulator.precision_at.items())},
        ndcg_at={k: statistics.mean(v) for k, v in sorted(accumulator.ndcg_at.items())},
        r_precision=(statistics.mean(accumulator.r_precision) if accumulator.r_precision else 0.0),
        recall_ceiling_at={
            k: statistics.mean(v) for k, v in sorted(accumulator.ceiling_at.items())
        },
        latency_ms={
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "mean": statistics.mean(latencies) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
        },
        unanswered=accumulator.unanswered,
        results_shown=(statistics.mean(accumulator.returned) if accumulator.returned else 0.0),
        depth_factor=depth_factor,
    )


def evaluate_engine(
    engine: Any,
    queries: Sequence[EvalQuery],
    *,
    name: str,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> MethodResult:
    """Score a search engine or a union retriever.

    A union retriever takes ``per_method`` rather than ``top_n`` and returns
    roughly twice as many documents, which is the whole point of it; scoring is
    done over everything it returns.
    """
    accumulator = _Accumulator()
    top_k = max(k_values)
    is_union = not hasattr(engine, "_index")
    # The union asks each retriever for k and returns their union, so its budget
    # is 2k. Scoring it at k would measure a retriever nobody ships.
    depth_factor = 2 if is_union else 1

    for item in queries:
        started = time.perf_counter()
        response = (
            engine.search(item.query, per_method=top_k)
            if is_union
            else engine.search(item.query, top_n=top_k)
        )
        accumulator.latencies.append((time.perf_counter() - started) * 1000.0)

        if response.is_empty:
            accumulator.unanswered += 1

        retrieved = [r.trial_id for r in response.results]
        _update(accumulator, retrieved, item.relevant, k_values, depth_factor)

    return _finalise(name, accumulator, len(queries), depth_factor)


def evaluate_baseline(
    baseline: Any,
    queries: Sequence[EvalQuery],
    preprocessor: Any,
    trial_ids: Sequence[str],
    *,
    name: str = "tfidf-baseline",
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> MethodResult:
    """Score the TF-IDF keyword baseline under identical conditions."""
    accumulator = _Accumulator()
    top_k = max(k_values)

    for item in queries:
        tokens = preprocessor.transform(item.query)
        started = time.perf_counter()
        hits = baseline.search(tokens, top_n=top_k)
        accumulator.latencies.append((time.perf_counter() - started) * 1000.0)

        if not hits:
            accumulator.unanswered += 1

        retrieved = [trial_ids[h.row_id] for h in hits]
        _update(accumulator, retrieved, item.relevant, k_values)

    return _finalise(name, accumulator, len(queries))


def run_evaluation(
    settings: Settings,
    *,
    eval_path: Path,
    field: FieldName = "abstract",
    models: Sequence[ModelName] = ("skipgram", "fasttext"),
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    include_baseline: bool = True,
    pooling: Pooling | None = None,
    include_union: bool = True,
) -> dict[str, Any]:
    """Evaluate every model plus the keyword baseline and write a report.

    Returns:
        The report payload, also written to ``reports/evaluation.json``.
    """
    from medsearch.data.loader import load_corpus
    from medsearch.pipelines.train import load_search_engine, run_preprocessing
    from medsearch.preprocessing.pipeline import TextPreprocessor
    from medsearch.search.baseline import TfidfBaseline

    queries = load_eval_set(eval_path)
    results: list[MethodResult] = []

    mode = pooling or settings.pooling
    for model in models:
        with stage(f"evaluate[{model}-{mode}]", logger):
            engine: Any = load_search_engine(settings, model, field, pooling=mode)
            # Warm up: the first query pays BLAS setup, which would otherwise
            # land entirely on the p99 of a 30-query set.
            engine.search("warm up query", top_n=max(k_values))
            label = model if mode == "mean" else f"{model}-{mode}"
            results.append(evaluate_engine(engine, queries, name=label, k_values=k_values))

    if include_union:
        from medsearch.pipelines.train import load_union_retriever

        for model in models:
            with stage(f"evaluate[union-{model}]", logger):
                union: Any = load_union_retriever(settings, model, field)
                union.search("warm up query")
                results.append(
                    evaluate_engine(union, queries, name=f"union-{model}", k_values=k_values)
                )

    if include_baseline:
        with stage("evaluate[tfidf-baseline]", logger):
            corpus = load_corpus(settings.paths.corpus_file)
            trial_ids = corpus["trial_id"].astype(str).tolist()
            cache, _, _ = run_preprocessing(settings, field)
            baseline = TfidfBaseline(cache)
            results.append(
                evaluate_baseline(
                    baseline,
                    queries,
                    TextPreprocessor(),
                    trial_ids,
                    k_values=k_values,
                )
            )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": str(settings.paths.corpus_file),
        "field": field,
        "pooling": mode,
        "eval_queries": len(queries),
        "k_values": list(k_values),
        "targets": {"recall_at_10": 0.70, "mrr_at_10": 0.45, "latency_p95_ms": 300.0},
        "results": [r.as_json() for r in results],
    }

    settings.paths.report_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.paths.report_dir / "evaluation.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", destination)

    return report


def check_targets(report: dict[str, Any]) -> list[str]:
    """Return a failure message per PRD target the best method misses.

    An empty list means every target is met.
    """
    embedding = [r for r in report["results"] if r["method"] != "tfidf-baseline"]
    if not embedding:
        return ["No embedding method was evaluated."]

    best = max(embedding, key=lambda r: r["recall_at"].get("10", 0.0))
    targets = report["targets"]
    failures: list[str] = []

    recall = float(best["recall_at"].get("10", 0.0))
    if recall < targets["recall_at_10"]:
        failures.append(
            f"Recall@10 {recall:.3f} is below the {targets['recall_at_10']} target "
            f"(best method: {best['method']}, "
            f"{float(best.get('results_shown', 0.0)):.1f} results per query)."
        )

    mrr = float(best["mrr_at"].get("10", 0.0))
    if mrr < targets["mrr_at_10"]:
        failures.append(
            f"MRR@10 {mrr:.3f} is below the {targets['mrr_at_10']} target "
            f"(best method: {best['method']})."
        )

    p95 = float(best["latency_ms"].get("p95", 0.0))
    if p95 > targets["latency_p95_ms"]:
        failures.append(
            f"p95 latency {p95:.1f} ms exceeds the {targets['latency_p95_ms']} ms target."
        )

    return failures
