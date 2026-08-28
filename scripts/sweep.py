"""Sprint 8.4 — hyperparameter sweep over the embedding training knobs.

Answers one question: **can tuning close the gap to TF-IDF?** The evaluation
(PRD §8.1) found the embeddings lose to a 40-line keyword baseline, and two
proposed remediations have already been measured and killed. Before accepting
that the dense representation is structurally outmatched, the obvious knobs
have to be ruled out rather than argued away.

Design decisions worth stating:

* **One-factor-at-a-time, not a grid.** A full cartesian product over four
  knobs is dozens of training runs to answer a screening question. OFAT around
  the shipped defaults finds a knob worth pursuing if one exists; if every
  single-factor move is flat, no interaction is going to rescue it.
* **Full corpus, not ``--limit``.** Phases.md 8.4 says to sweep with a row cap,
  written before Sprint 8 measured the real cost. Training is 30 s and peaks at
  319 MB (Architecture.md §9), so the cap buys nothing and would break the
  measurement: the eval set's relevant documents mostly fall outside a 2 000-row
  sample, so recall on a subset measures the sample, not the model.
* **Skip-gram, not FastText.** The two are statistically indistinguishable
  (p = 0.28 / 0.86 / 1.00, PRD §8.1) and Skip-gram trains in half the time.
* **Scratch artefacts.** Each config trains into a temporary directory so the
  shipped models and indexes in ``models/`` and ``data/processed/`` survive the
  sweep untouched.

Both the embedding ranker and the union retriever are scored per config: the
union is what ships, so a knob that helps only the standalone ranker is not
actually a win.

Run: ``python scripts/sweep.py`` -> ``reports/sweep.json``

The 2 GB training floor is the binding constraint on a busy machine, not the
346 MB the run actually peaks at. With less than 2 GB free, prefix the run with
``MEDSEARCH_MIN_FREE_MEMORY_GB=1.0 MEDSEARCH_WARN_FREE_MEMORY_GB=1.0`` -- still
3x the measured need, and the sweep is how this was run.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from medsearch.config import Settings, get_settings
from medsearch.logging_conf import configure_logging, get_logger
from medsearch.pipelines.evaluate import evaluate_engine, load_eval_set
from medsearch.pipelines.train import build_index, load_union_retriever, train_one

logger = get_logger(__name__)

MODEL = "skipgram"
FIELD = "abstract"
K = 10

#: The shipped defaults. Every config below is this dict with one key changed,
#: so any difference is attributable to that one knob.
BASE: dict[str, int] = {"vector_size": 100, "window": 5, "min_count": 2, "epochs": 5}

#: Alternative values per knob. Chosen to bracket the default rather than to
#: fine-tune around it -- this is a screen, so the moves are large enough that
#: a real effect would be visible.
VARIANTS: dict[str, tuple[int, ...]] = {
    "vector_size": (200, 300),
    "window": (10,),
    "min_count": (5,),
    "epochs": (15,),
}


@dataclass(frozen=True, slots=True)
class SweepResult:
    """One configuration's scores."""

    label: str
    params: dict[str, int]
    recall_at_10: float
    mrr_at_10: float
    union_recall_at_10: float
    union_mrr_at_10: float
    train_seconds: float


def _configs() -> list[tuple[str, dict[str, int]]]:
    """Base config first, then one entry per single-knob change."""
    out: list[tuple[str, dict[str, int]]] = [("baseline", dict(BASE))]
    for knob, values in VARIANTS.items():
        for value in values:
            out.append((f"{knob}={value}", {**BASE, knob: value}))
    return out


def _scratch(source: Settings, root: Path) -> tuple[Path, Path]:
    """Copy the inputs a training run reads into an isolated tree.

    Only ``raw`` (the corpus) and ``interim`` (the token cache) are copied.
    Leaving ``processed`` empty is deliberate: it is where indexes land, and an
    empty one guarantees each config builds its own rather than silently
    scoring a stale neighbour's.
    """
    data = root / "data"
    (data / "processed").mkdir(parents=True)
    shutil.copytree(source.paths.raw_dir, data / "raw")
    if source.paths.interim_dir.exists():
        shutil.copytree(source.paths.interim_dir, data / "interim")
    models = root / "models"
    models.mkdir()
    return data, models


def _score(settings: Settings, queries: list[Any], label: str) -> tuple[float, ...]:
    """Train, index, and score one configuration."""
    outcome = train_one(settings, cast(Any, MODEL), cast(Any, FIELD))
    build_index(settings, cast(Any, MODEL), cast(Any, FIELD))

    from medsearch.pipelines.train import load_search_engine

    engine = load_search_engine(settings, cast(Any, MODEL), cast(Any, FIELD))
    solo = evaluate_engine(engine, queries, name=label, k_values=(K,))

    union = load_union_retriever(settings, cast(Any, MODEL), cast(Any, FIELD))
    fused = evaluate_engine(union, queries, name=f"union-{label}", k_values=(K,))

    return (
        solo.recall_at[K],
        solo.mrr_at[K],
        fused.recall_at[K],
        fused.mrr_at[K],
        outcome.seconds,
    )


def main() -> None:
    base_settings = get_settings()
    configure_logging(base_settings.log_level)
    queries = load_eval_set(Path("tests/fixtures/eval_queries.json"))
    logger.info("Sweeping %d configs over %d queries", len(_configs()), len(queries))

    results: list[SweepResult] = []
    with tempfile.TemporaryDirectory(prefix="medsearch-sweep-") as tmp:
        data_dir, model_dir = _scratch(base_settings, Path(tmp))
        for label, params in _configs():
            settings = base_settings.model_copy(
                update={"data_dir": data_dir, "model_dir": model_dir, **params}
            )
            recall, mrr, u_recall, u_mrr, seconds = _score(settings, queries, label)
            results.append(SweepResult(label, params, recall, mrr, u_recall, u_mrr, seconds))
            # print, not log: the sweep is long enough that a suppressed log
            # level would lose every intermediate result if the run dies late.
            print(f"  {label:<16} R@10 {recall:.4f}   union R@10 {u_recall:.4f}", flush=True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": MODEL,
        "field": FIELD,
        "eval_queries": len(queries),
        "note": (
            "One-factor-at-a-time around the shipped defaults, full corpus. "
            "See PRD section 8 for why the sweep was run at all."
        ),
        "results": [asdict(r) for r in results],
    }
    out = base_settings.paths.report_dir / "sweep.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    baseline = results[0]
    print(f"\nSweep over {len(results)} configs, {len(queries)} queries\n")
    for r in results:
        delta = r.recall_at_10 - baseline.recall_at_10
        u_delta = r.union_recall_at_10 - baseline.union_recall_at_10
        print(
            f"  {r.label:<16} R@10 {r.recall_at_10:.4f} ({delta:+.4f})"
            f"   union {r.union_recall_at_10:.4f} ({u_delta:+.4f})"
        )
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
