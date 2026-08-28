"""Artefact integrity checks.

Covers the third artefact-mismatch class, found in Sprint 11: an index built
from a corpus that has since been replaced. Row ids are positional, so such an
index still resolves — to the wrong documents — which makes it more dangerous
than the two mismatch classes already guarded.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from medsearch.config import Settings
from medsearch.embeddings.base import ModelMetadata
from medsearch.embeddings.registry import metadata_path, vectors_path
from medsearch.pipelines.integrity import (
    IntegrityIssue,
    Severity,
    check_artefacts,
)
from medsearch.search.index import DocumentIndex


def install_model(
    settings: Settings,
    model: str,
    *,
    fingerprint: str = "model-fp",
    corpus_fingerprint: str = "corpus-fp",
    documents: int = 20,
    sampled: bool = False,
    artefact_bytes: int = 2048,
) -> None:
    """Write a minimal but structurally valid model artefact."""
    directory = settings.paths.model_path(model)  # type: ignore[arg-type]
    directory.mkdir(parents=True, exist_ok=True)
    vectors_path(directory).write_bytes(b"x" * artefact_bytes)
    ModelMetadata(
        kind=model,
        fingerprint=fingerprint,
        corpus_fingerprint=corpus_fingerprint,
        corpus_documents=documents,
        vocabulary_size=100,
        params={},
        gensim_version="4.4.0",
        artefact_bytes=artefact_bytes,
        training_seconds=1.0,
        sampled=sampled,
    ).save(metadata_path(directory))


def install_index(
    settings: Settings,
    model: str,
    field: str = "abstract",
    *,
    model_fingerprint: str = "model-fp",
    corpus_fingerprint: str = "corpus-fp",
    rows: int = 20,
) -> None:
    """Write a matching index artefact."""
    DocumentIndex(
        vectors=np.eye(rows, 4, dtype=np.float32),
        row_ids=np.arange(rows, dtype=np.int64),
        model_fingerprint=model_fingerprint,
        model_kind=model,
        field=field,
        corpus_fingerprint=corpus_fingerprint,
    ).save(settings.paths.index_path(model, field))  # type: ignore[arg-type]


def codes(issues: list[IntegrityIssue]) -> set[str]:
    return {issue.code for issue in issues}


@pytest.fixture
def live_fingerprint(settings: Settings) -> str:
    from medsearch.data.loader import corpus_fingerprint

    return corpus_fingerprint(settings.paths.corpus_file)


class TestCleanState:
    def test_consistent_artefacts_produce_no_findings(
        self, settings: Settings, live_fingerprint: str
    ) -> None:
        for model in ("skipgram", "fasttext"):
            install_model(settings, model, corpus_fingerprint=live_fingerprint)
            install_index(settings, model, corpus_fingerprint=live_fingerprint)
        assert check_artefacts(settings) == []


class TestMissingArtefacts:
    def test_untrained_models_are_reported(self, settings: Settings) -> None:
        assert codes(check_artefacts(settings)) == {"model-missing"}

    def test_untrained_is_a_warning_not_an_error(self, settings: Settings) -> None:
        assert all(i.severity is Severity.WARN for i in check_artefacts(settings))

    def test_missing_corpus_short_circuits(self, settings: Settings) -> None:
        settings.paths.corpus_file.unlink()
        issues = check_artefacts(settings)
        assert codes(issues) == {"corpus-missing"}
        assert issues[0].severity is Severity.ERROR

    def test_model_without_an_index_is_not_an_error(
        self, settings: Settings, live_fingerprint: str
    ) -> None:
        install_model(settings, "skipgram", corpus_fingerprint=live_fingerprint)
        assert "index-corpus-stale" not in codes(check_artefacts(settings))


class TestStaleCorpus:
    """The Sprint 11 finding: the corpus changed after the index was built."""

    def test_stale_index_is_an_error(self, settings: Settings, live_fingerprint: str) -> None:
        install_model(settings, "skipgram", corpus_fingerprint=live_fingerprint)
        install_index(settings, "skipgram", corpus_fingerprint="an-older-corpus")
        issues = [i for i in check_artefacts(settings) if i.code == "index-corpus-stale"]
        assert len(issues) == 1
        assert issues[0].severity is Severity.ERROR

    def test_stale_message_explains_the_danger(
        self, settings: Settings, live_fingerprint: str
    ) -> None:
        install_model(settings, "skipgram", corpus_fingerprint=live_fingerprint)
        install_index(settings, "skipgram", corpus_fingerprint="an-older-corpus")
        message = next(
            i.message for i in check_artefacts(settings) if i.code == "index-corpus-stale"
        )
        assert "WRONG documents" in message
        assert "positional" in message

    def test_stale_model_is_a_warning(self, settings: Settings, live_fingerprint: str) -> None:
        # A stale *model* still produces coherent vectors; only a stale index
        # mis-resolves documents. Severity reflects that difference.
        install_model(settings, "skipgram", corpus_fingerprint="an-older-corpus")
        install_index(settings, "skipgram", corpus_fingerprint=live_fingerprint)
        issue = next(i for i in check_artefacts(settings) if i.code == "model-corpus-stale")
        assert issue.severity is Severity.WARN

    def test_errors_sort_before_warnings(self, settings: Settings) -> None:
        install_model(settings, "skipgram", corpus_fingerprint="older")
        install_index(settings, "skipgram", corpus_fingerprint="older")
        issues = check_artefacts(settings)
        severities = [i.severity for i in issues]
        assert severities == sorted(severities, key=lambda s: 0 if s is Severity.ERROR else 1)


class TestModelIndexMismatch:
    def test_index_from_a_different_model_is_an_error(
        self, settings: Settings, live_fingerprint: str
    ) -> None:
        install_model(settings, "skipgram", fingerprint="fp-a", corpus_fingerprint=live_fingerprint)
        install_index(
            settings, "skipgram", model_fingerprint="fp-b", corpus_fingerprint=live_fingerprint
        )
        issue = next(i for i in check_artefacts(settings) if i.code == "index-model-mismatch")
        assert issue.severity is Severity.ERROR


class TestSampledArtefacts:
    def test_sampled_model_is_flagged(self, settings: Settings, live_fingerprint: str) -> None:
        install_model(
            settings, "skipgram", corpus_fingerprint=f"{live_fingerprint}-n10", sampled=True
        )
        assert "model-sampled" in codes(check_artefacts(settings))

    def test_sampled_index_is_flagged_not_treated_as_stale(
        self, settings: Settings, live_fingerprint: str
    ) -> None:
        # A `-n10` suffix is expected on a sampled index and must not be
        # mistaken for a corpus change.
        install_model(
            settings, "skipgram", corpus_fingerprint=f"{live_fingerprint}-n10", sampled=True
        )
        install_index(settings, "skipgram", corpus_fingerprint=f"{live_fingerprint}-n10", rows=10)
        found = codes(check_artefacts(settings))
        assert "index-sampled" in found
        assert "index-corpus-stale" not in found


class TestSizeMismatch:
    def test_row_count_disagreement_is_an_error(
        self, settings: Settings, live_fingerprint: str
    ) -> None:
        install_model(settings, "skipgram", corpus_fingerprint=live_fingerprint)
        install_index(settings, "skipgram", corpus_fingerprint=live_fingerprint, rows=7)
        issue = next(i for i in check_artefacts(settings) if i.code == "index-size-mismatch")
        assert issue.severity is Severity.ERROR


class TestOversizedArtefact:
    def test_over_budget_model_is_flagged(self, settings: Settings, live_fingerprint: str) -> None:
        install_model(
            settings,
            "skipgram",
            corpus_fingerprint=live_fingerprint,
            artefact_bytes=200 * 1024**2,
        )
        assert "artefact-oversized" in codes(check_artefacts(settings))


class TestRendering:
    def test_render_includes_severity_and_code(self) -> None:
        rendered = IntegrityIssue(Severity.ERROR, "some-code", "a message").render()
        assert "ERROR" in rendered
        assert "[some-code]" in rendered

    def test_severity_stringifies_cleanly(self) -> None:
        assert str(Severity.WARN) == "WARN"


@pytest.mark.slow
class TestStaleIndexErrorAtLoad:
    """`load_search_engine` must refuse a stale index outright.

    Marked slow: trains a real model to produce a genuine index to corrupt.
    """

    def test_load_raises_stale_index_error(self, settings: Settings, live_fingerprint: str) -> None:
        from medsearch.exceptions import StaleIndexError
        from medsearch.pipelines.train import build_index, load_search_engine, train_one

        train_one(settings, "skipgram", "abstract")
        build_index(settings, "skipgram", "abstract")

        manifest = settings.paths.index_path("skipgram", "abstract") / "manifest.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload["corpus_fingerprint"] = "a-different-corpus"
        manifest.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(StaleIndexError, match="stale"):
            load_search_engine(settings, "skipgram", "abstract")

    def test_error_names_both_fingerprints(self, settings: Settings) -> None:
        from medsearch.exceptions import StaleIndexError

        error = StaleIndexError(expected="new-fp", actual="old-fp", documents=10_666)
        message = str(error)
        assert "new-fp" in message
        assert "old-fp" in message
        assert "10,666" in message
