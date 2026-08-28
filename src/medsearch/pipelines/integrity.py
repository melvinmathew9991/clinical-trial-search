"""Artefact integrity checks.

Every artefact carries a fingerprint of what produced it. This module is where
those fingerprints are actually *compared* — against the model, against the
live corpus, and against each other — so a mismatch is reported rather than
silently served.

Three mismatch classes exist, and the project found them in this order:

1. **Model ↔ index.** The legacy project wrote Skip-gram vectors into files
   named FastText. Caught by ``DocumentIndex.load(expected_fingerprint=...)``.
2. **Sampled index vs. full corpus.** A 2,000-vector development index paired
   silently with all 10,666 corpus rows. Caught by ``DocumentIndex.is_sampled``.
3. **Stale index after the corpus changed.** Found during Sprint 11 and fixed
   here. This is the dangerous one: row ids are *positional*, so if the corpus
   is replaced the index still resolves — to the wrong documents. A result
   would show one trial's title beside another trial's relevance score.

Class 3 is not hypothetical. The Azure pipeline is triggered by a new CSV
landing in blob storage; if training fails after the drop but the app
restarts, it serves an index built from the previous corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from medsearch.config import FieldName, ModelName, Settings
from medsearch.data.loader import corpus_fingerprint
from medsearch.embeddings.base import ModelKind
from medsearch.embeddings.registry import is_trained, load_metadata
from medsearch.logging_conf import get_logger
from medsearch.search.index import DocumentIndex

logger = get_logger(__name__)

ALL_MODELS: tuple[ModelName, ...] = ("skipgram", "fasttext")
ALL_FIELDS: tuple[FieldName, ...] = ("abstract", "title")


class Severity(str, Enum):
    """How much a finding matters."""

    ERROR = "ERROR"
    WARN = "WARN"
    OK = "OK"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class IntegrityIssue:
    """One finding from an artefact check."""

    severity: Severity
    code: str
    message: str

    def render(self) -> str:
        """Single-line rendering for the CLI."""
        return f"  {self.severity.value:<5} [{self.code}] {self.message}"


def _check_index(
    settings: Settings,
    model: ModelName,
    field: FieldName,
    model_fingerprint: str,
    live_corpus_fingerprint: str,
    corpus_rows: int,
) -> list[IntegrityIssue]:
    """Verify one index against its model and the live corpus."""
    directory = settings.paths.index_path(model, field)
    issues: list[IntegrityIssue] = []

    if not DocumentIndex.exists(directory):
        return issues  # not every model/field combination has to be built

    index = DocumentIndex.load(directory, mmap=True)
    try:
        label = f"{model}-{field}"

        if index.model_fingerprint != model_fingerprint:
            issues.append(
                IntegrityIssue(
                    Severity.ERROR,
                    "index-model-mismatch",
                    f"{label}: index was built by model {index.model_fingerprint}, "
                    f"but the installed model is {model_fingerprint}. "
                    f"Rebuild: medsearch index build --model {model}",
                )
            )

        expected_corpus = (
            f"{live_corpus_fingerprint}-n{index.sampled_limit}"
            if index.is_sampled
            else live_corpus_fingerprint
        )
        if index.corpus_fingerprint != expected_corpus:
            issues.append(
                IntegrityIssue(
                    Severity.ERROR,
                    "index-corpus-stale",
                    f"{label}: index was built from corpus {index.corpus_fingerprint}, "
                    f"but data/raw now holds {expected_corpus}. Row ids are positional, "
                    f"so this index would resolve to the WRONG documents. "
                    f"Rebuild: medsearch train && medsearch index build",
                )
            )

        if index.is_sampled:
            issues.append(
                IntegrityIssue(
                    Severity.WARN,
                    "index-sampled",
                    f"{label}: development index covering {index.size:,} of "
                    f"{corpus_rows:,} documents. Rebuild without --limit for full coverage.",
                )
            )
        elif index.size != corpus_rows:
            issues.append(
                IntegrityIssue(
                    Severity.ERROR,
                    "index-size-mismatch",
                    f"{label}: index holds {index.size:,} vectors but the corpus has "
                    f"{corpus_rows:,} rows.",
                )
            )
    finally:
        index.close()

    return issues


def check_artefacts(settings: Settings) -> list[IntegrityIssue]:
    """Verify every model and index against the live corpus.

    Args:
        settings: Configuration naming the corpus and artefact directories.

    Returns:
        Findings, most severe first. An empty list means everything on disk is
        internally consistent and current.
    """
    issues: list[IntegrityIssue] = []

    corpus_file = settings.paths.corpus_file
    if not corpus_file.exists():
        return [
            IntegrityIssue(
                Severity.ERROR,
                "corpus-missing",
                f"No corpus at {corpus_file}. Nothing can be verified against it.",
            )
        ]

    live_fingerprint = corpus_fingerprint(corpus_file)
    corpus_rows = _count_corpus_rows(settings)

    for model in ALL_MODELS:
        model_dir = settings.paths.model_path(model)

        if not is_trained(model_dir):
            issues.append(
                IntegrityIssue(
                    Severity.WARN,
                    "model-missing",
                    f"{model}: not trained. Run: medsearch train --model {model}",
                )
            )
            continue

        metadata = load_metadata(model_dir, ModelKind(model))

        if metadata.sampled:
            issues.append(
                IntegrityIssue(
                    Severity.WARN,
                    "model-sampled",
                    f"{model}: trained on a {metadata.corpus_documents:,}-document sample. "
                    f"Development artefact — not for production.",
                )
            )

        base_fingerprint = metadata.corpus_fingerprint.split("-n")[0]
        if base_fingerprint != live_fingerprint:
            issues.append(
                IntegrityIssue(
                    Severity.WARN,
                    "model-corpus-stale",
                    f"{model}: trained on corpus {base_fingerprint}, but data/raw now "
                    f"holds {live_fingerprint}. Retrain to pick up the new data.",
                )
            )

        if metadata.artefact_mb > settings.max_artefact_mb:
            issues.append(
                IntegrityIssue(
                    Severity.WARN,
                    "artefact-oversized",
                    f"{model}: {metadata.artefact_mb:.0f} MB exceeds the "
                    f"{settings.max_artefact_mb} MB budget (ADR-001).",
                )
            )

        for field in ALL_FIELDS:
            issues.extend(
                _check_index(
                    settings, model, field, metadata.fingerprint, live_fingerprint, corpus_rows
                )
            )

    order = {Severity.ERROR: 0, Severity.WARN: 1, Severity.OK: 2}
    issues.sort(key=lambda i: order[i.severity])
    return issues


def _count_corpus_rows(settings: Settings) -> int:
    """Row count of the live corpus, without holding the frame."""
    from medsearch.data.loader import load_corpus

    frame = load_corpus(settings.paths.corpus_file)
    count = len(frame)
    del frame
    return count


def verify_or_warn(settings: Settings, model: ModelName, field: FieldName) -> list[IntegrityIssue]:
    """Check one model/field pair, logging anything found.

    Used at serving time, where a stale index must not pass unnoticed.
    """
    relevant = [
        issue
        for issue in check_artefacts(settings)
        if f"{model}-{field}" in issue.message or issue.message.startswith(f"{model}:")
    ]
    for issue in relevant:
        if issue.severity is Severity.ERROR:
            logger.error("%s: %s", issue.code, issue.message)
        else:
            logger.warning("%s: %s", issue.code, issue.message)
    return relevant
