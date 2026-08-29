"""Regression tests for the four defects found in the legacy reference/legacy code.

Each test here exists because the original project shipped the bug. They are
grouped in one module so the link between "what went wrong before" and "what
now prevents it" stays visible (Rules.md section 5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from medsearch.config import Settings
from medsearch.data.loader import load_corpus
from medsearch.embeddings.base import ModelKind, TrainingParams
from medsearch.embeddings.document import DocumentEmbedder, l2_normalize
from medsearch.exceptions import ArtefactMismatchError, SchemaValidationError
from medsearch.search.index import DocumentIndex


class TestHiddenRowLimit:
    """Legacy ``utils.py:8`` returned ``df.iloc[:100, :]`` unconditionally.

    Every "production" model the original project shipped was therefore
    trained on 100 of 10,666 abstracts, silently (ADR-007).
    """

    def test_load_corpus_returns_all_rows_by_default(self, corpus_csv: Path) -> None:
        frame = load_corpus(corpus_csv)
        assert len(frame) == 20, "the full corpus must be returned when limit is None"

    def test_limit_is_opt_in_and_respected(self, corpus_csv: Path) -> None:
        frame = load_corpus(corpus_csv, limit=2)
        assert len(frame) == 2

    def test_limit_emits_a_warning(
        self, corpus_csv: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            load_corpus(corpus_csv, limit=2)
        assert "SAMPLED CORPUS" in caplog.text


class TestArtefactMismatch:
    """Legacy ``engine.py:36,41`` wrote Skip-gram vectors into the FastText files.

    Verified against the shipped artefacts: ``FastText-vec-abstract.csv`` was
    byte-for-byte the transpose of ``skipgram-vec-abstract.csv``. Nothing in
    the code could detect it. The fingerprint stamp now can.
    """

    def _index(self, tmp_path: Path, fingerprint: str) -> Path:
        directory = tmp_path / "idx"
        DocumentIndex(
            vectors=np.eye(4, dtype=np.float32),
            row_ids=np.arange(4),
            model_fingerprint=fingerprint,
            model_kind="skipgram",
            field="abstract",
            corpus_fingerprint="corpus1",
        ).save(directory)
        return directory

    def test_matching_fingerprint_loads(self, tmp_path: Path) -> None:
        directory = self._index(tmp_path, "skipgram-fp")
        with DocumentIndex.load(directory, expected_fingerprint="skipgram-fp") as index:
            assert index.size == 4

    def test_wrong_model_is_rejected(self, tmp_path: Path) -> None:
        directory = self._index(tmp_path, "skipgram-fp")
        with pytest.raises(ArtefactMismatchError, match="fingerprint mismatch"):
            DocumentIndex.load(directory, expected_fingerprint="fasttext-fp")


class TestSchemaValidation:
    """Legacy code indexed columns positionally (``df1.iloc[L, [1,2,5,6]]``).

    A reordered export would have returned the wrong fields with no error.
    """

    def test_missing_required_column_names_the_offender(
        self, corpus_missing_abstract: Path
    ) -> None:
        with pytest.raises(SchemaValidationError) as exc_info:
            load_corpus(corpus_missing_abstract)
        assert "Abstract" in str(exc_info.value)

    def test_columns_are_renamed_to_snake_case(self, corpus_csv: Path) -> None:
        frame = load_corpus(corpus_csv)
        assert "trial_id" in frame.columns
        assert "publication_date" in frame.columns
        assert "Trial ID" not in frame.columns


class TestNoMutationOfCallerFrame:
    """Legacy ``output_text`` did ``df[column_name][i] = ...`` in a loop.

    Chained assignment, quadratic copying, and it mutated the caller's frame
    in place (PRD F-10).
    """

    def test_preprocessing_leaves_the_source_frame_untouched(self, corpus_csv: Path) -> None:
        frame = load_corpus(corpus_csv)
        before = frame["abstract"].tolist()

        from medsearch.preprocessing.normalizer import clean_text

        cleaned = [clean_text(text) for text in frame["abstract"]]

        # Values, not dtype: the loader deliberately uses pandas StringDtype.
        assert frame["abstract"].tolist() == before
        assert cleaned != before, "the transform must actually change the text"

    def test_clean_text_returns_a_new_string(self) -> None:
        from medsearch.preprocessing.normalizer import clean_text

        original = "COVID-19 Patients (n=1200)"
        assert clean_text(original) != original
        assert original == "COVID-19 Patients (n=1200)"


class TestZeroVectorHandling:
    """Legacy ``get_mean_vector`` returned ``np.array([0]*100)`` for an all-OOV
    document, which then produced ``0/0 -> nan`` inside ``cos_sim`` and ranked
    as a silent NaN.
    """

    def test_all_oov_document_gets_a_zero_vector_not_a_crash(self, fake_vectors: object) -> None:
        embedder = DocumentEmbedder(fake_vectors)
        vector = embedder.embed(["nonexistentword", "anotherfakeword"])
        assert vector.shape == (4,)
        assert not np.isnan(vector).any()
        assert np.count_nonzero(vector) == 0
        assert embedder.oov_documents == 1

    def test_l2_normalize_never_produces_nan_from_zero_rows(self) -> None:
        matrix = np.array([[3, 4, 0, 0], [0, 0, 0, 0]], dtype=np.float32)
        normalised = l2_normalize(matrix)
        assert not np.isnan(normalised).any()
        assert np.isclose(np.linalg.norm(normalised[0]), 1.0)
        assert np.isclose(np.linalg.norm(normalised[1]), 0.0)


class TestFastTextBucketBound:
    """Legacy training left ``bucket`` at gensim's 2,000,000 default.

    At 100 dims x float32 that is exactly 800,000,000 bytes -- the size of the
    shipped ``model_Fasttext.bin.wv.vectors_ngrams.npy`` (ADR-001).
    """

    def test_gensim_default_bucket_predicts_the_legacy_800mb_artefact(self) -> None:
        legacy = TrainingParams(vector_size=100, bucket=2_000_000)
        assert legacy.ngram_matrix_bytes() == 800_000_000

    def test_project_default_is_bounded(self) -> None:
        params = TrainingParams(vector_size=100)
        assert params.ngram_matrix_bytes() == 20_000_000
        assert params.ngram_matrix_bytes() < 150 * 1024**2

    def test_settings_rejects_a_bucket_above_the_gensim_default(self) -> None:
        with pytest.raises(ValueError, match="less than or equal to"):
            Settings(fasttext_bucket=5_000_000)


class TestWorkerCount:
    """Legacy training used ``workers=5`` on a 4-logical-core machine."""

    def test_auto_workers_leaves_one_core_free(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("medsearch.config.os.cpu_count", lambda: 4)
        assert Settings(workers=0).effective_workers == 3

    def test_auto_workers_never_returns_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("medsearch.config.os.cpu_count", lambda: 1)
        assert Settings(workers=0).effective_workers == 1


class TestScaledMemoryFloor:
    """The memory floor was flat, so `--limit 2000` was blocked by the
    full-corpus requirement -- while the ResourceError message recommended
    exactly that flag as the remedy. Found during Track 0 on a machine with
    0.79 GB free. The floor now scales with the documents actually processed.
    """

    def test_full_corpus_uses_the_configured_floor(self) -> None:
        settings = Settings(min_free_memory_gb=2.0)
        assert settings.memory_floor_gb(None) == 2.0

    def test_sampled_run_scales_down(self) -> None:
        settings = Settings(min_free_memory_gb=2.0)
        assert settings.memory_floor_gb(2000) < settings.memory_floor_gb(None)

    def test_sampled_floor_never_drops_below_the_absolute_minimum(self) -> None:
        settings = Settings(min_free_memory_gb=2.0)
        assert settings.memory_floor_gb(1) == pytest.approx(0.5)

    def test_floor_never_exceeds_the_configured_maximum(self) -> None:
        settings = Settings(min_free_memory_gb=2.0)
        assert settings.memory_floor_gb(999_999) <= 2.0

    def test_the_advertised_fallback_is_actually_permitted(self) -> None:
        # The exact scenario that failed: 0.79 GB free, --limit 2000.
        settings = Settings(min_free_memory_gb=2.0)
        assert settings.memory_floor_gb(2000) <= 0.79

    def test_remedy_text_does_not_suggest_limit_when_already_sampled(self) -> None:
        from medsearch.exceptions import ResourceError
        from medsearch.runtime import require_memory

        with pytest.raises(ResourceError) as exc_info:
            require_memory(999.0, stage="train:skipgram", limit=2000)
        message = str(exc_info.value)
        assert "already sampled" in message
        assert "run a reduced profile with `--limit 2000`" not in message


class TestFastTextMemoryGuardScales:
    """The same flat-floor defect as :class:`TestScaledMemoryFloor`, in a second
    place that was missed.

    ``runtime.require_memory`` learned to scale its floor with the documents
    actually processed; ``trainer._check_memory_budget`` kept a bare ``+ 0.5``
    and so demanded as much free RAM to train 20 documents as to train 10,666.
    That is what made ``TestCrossModelGuard`` intermittent -- it failed only
    when the machine happened to be under memory pressure, which looked like
    test-order dependence and was not.

    The message was unreadable too: the headroom was invisible, so a refusal
    printed as "peak is ~0.00 GB but only 0.42 GB is available", which reads as
    an arithmetic bug rather than a memory limit.
    """

    @staticmethod
    def _budget(free_gb: float, documents: int, monkeypatch: pytest.MonkeyPatch) -> None:
        from medsearch.embeddings import trainer

        monkeypatch.setattr(trainer, "available_memory_gb", lambda: free_gb)
        trainer._check_memory_budget(ModelKind.FASTTEXT, TrainingParams(), documents)

    def test_a_toy_corpus_needs_less_than_the_full_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 0.42 GB free is the figure CI actually failed at.
        self._budget(0.42, 20, monkeypatch)

    def test_the_full_corpus_still_enforces_a_real_floor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from medsearch.exceptions import ResourceError

        with pytest.raises(ResourceError):
            self._budget(0.42, 10_666, monkeypatch)

    def test_unknown_document_count_budgets_the_full_corpus(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Zero means unknown, and must fail safe rather than permissive."""
        from medsearch.exceptions import ResourceError

        with pytest.raises(ResourceError):
            self._budget(0.42, 0, monkeypatch)

    def test_the_message_shows_where_the_requirement_comes_from(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from medsearch.exceptions import ResourceError

        with pytest.raises(ResourceError) as exc_info:
            self._budget(0.10, 10_666, monkeypatch)
        message = str(exc_info.value)
        assert "n-gram matrix" in message
        assert "headroom" in message
        assert "10,666 documents" in message

    def test_skipgram_is_not_subject_to_the_fasttext_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from medsearch.embeddings import trainer

        monkeypatch.setattr(trainer, "available_memory_gb", lambda: 0.0)
        trainer._check_memory_budget(ModelKind.SKIPGRAM, TrainingParams(), 10_666)


class TestSampledIndexIsDeclared:
    """A sampled index paired silently with the full corpus.

    Found in Track 0: the app loaded a 2,000-vector index alongside all 10,666
    corpus rows. Results were correct, but 8,666 trials were unreachable and
    nothing said so -- the same silent-mispairing class as the legacy
    Skipgram/FastText artefact swap.
    """

    def _index(self, corpus_fingerprint: str) -> DocumentIndex:
        return DocumentIndex(
            vectors=np.eye(3, dtype=np.float32),
            row_ids=np.arange(3),
            model_fingerprint="fp",
            model_kind="skipgram",
            field="abstract",
            corpus_fingerprint=corpus_fingerprint,
        )

    def test_sampled_limit_is_recovered_from_the_fingerprint(self) -> None:
        assert self._index("abc123-n2000").sampled_limit == 2000

    def test_full_corpus_index_reports_no_limit(self) -> None:
        assert self._index("abc123").sampled_limit is None

    def test_is_sampled_flag(self) -> None:
        assert self._index("abc123-n500").is_sampled is True
        assert self._index("abc123").is_sampled is False

    def test_a_hex_fingerprint_is_not_mistaken_for_a_limit(self) -> None:
        # 'n' followed by non-digits must not parse as a sample marker.
        assert self._index("deadbeefncafe").sampled_limit is None


class TestStaleIndexAfterRetrain:
    """A retrain with identical hyperparameters must invalidate its index.

    Found during the pre-deployment domain audit. ``fingerprint`` hashes
    (kind, corpus, hyperparameters) -- not the vectors -- so two training runs
    of the same configuration share a fingerprint. gensim is not deterministic
    above one worker, so those two runs hold *different* vectors. An index
    built from the first therefore paired "validly" with the second: the
    fingerprint matched, ``doctor --full`` reported every artefact consistent,
    and the stored rows scored cosine 0.96 against the live model rather than
    1.00.

    That is the legacy K1/K2 failure exactly -- wrong pairing, no signal --
    which the fingerprint was introduced to prevent. These pin the fix.
    """

    def test_checksum_differs_for_different_vectors(self) -> None:
        from medsearch.embeddings.base import vectors_checksum

        first = np.arange(12, dtype=np.float32).reshape(3, 4)
        second = first.copy()
        second[0, 0] += 0.001
        assert vectors_checksum(first) != vectors_checksum(second)

    def test_checksum_is_stable_for_identical_vectors(self) -> None:
        from medsearch.embeddings.base import vectors_checksum

        matrix = np.arange(12, dtype=np.float32).reshape(3, 4)
        assert vectors_checksum(matrix) == vectors_checksum(matrix.copy())

    def test_checksum_distinguishes_shape(self) -> None:
        """Same bytes, different interpretation, must not collide."""
        from medsearch.embeddings.base import vectors_checksum

        flat = np.arange(12, dtype=np.float32)
        assert vectors_checksum(flat.reshape(3, 4)) != vectors_checksum(flat.reshape(4, 3))

    def test_retrained_model_is_rejected_by_its_old_index(self, tmp_path: Path) -> None:
        """The scenario that slipped through: same fingerprint, new vectors."""
        from medsearch.exceptions import ArtefactMismatchError
        from medsearch.search.index import DocumentIndex

        index = DocumentIndex(
            vectors=np.eye(3, dtype=np.float32),
            row_ids=np.arange(3, dtype=np.int64),
            model_fingerprint="same-config",
            model_kind="skipgram",
            field="abstract",
            corpus_fingerprint="corpus",
            model_vectors_checksum="vectors-from-run-1",
        )
        index.save(tmp_path / "idx")

        with pytest.raises(ArtefactMismatchError):
            DocumentIndex.load(
                tmp_path / "idx",
                expected_fingerprint="same-config",  # unchanged -- this is the point
                expected_vectors_checksum="vectors-from-run-2",
            )

    def test_matching_checksum_loads(self, tmp_path: Path) -> None:
        from medsearch.search.index import DocumentIndex

        index = DocumentIndex(
            vectors=np.eye(3, dtype=np.float32),
            row_ids=np.arange(3, dtype=np.int64),
            model_fingerprint="same-config",
            model_kind="skipgram",
            field="abstract",
            corpus_fingerprint="corpus",
            model_vectors_checksum="vectors-from-run-1",
        )
        index.save(tmp_path / "idx")
        loaded = DocumentIndex.load(
            tmp_path / "idx",
            expected_fingerprint="same-config",
            expected_vectors_checksum="vectors-from-run-1",
        )
        assert loaded.size == 3

    def test_legacy_index_without_a_checksum_warns_but_loads(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Indexes written before the field existed must stay loadable."""
        import json
        import logging

        from medsearch.search.index import DocumentIndex

        index = DocumentIndex(
            vectors=np.eye(3, dtype=np.float32),
            row_ids=np.arange(3, dtype=np.int64),
            model_fingerprint="same-config",
            model_kind="skipgram",
            field="abstract",
            corpus_fingerprint="corpus",
        )
        index.save(tmp_path / "idx")
        manifest_file = tmp_path / "idx" / "manifest.json"
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        del manifest["model_vectors_checksum"]
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            loaded = DocumentIndex.load(
                tmp_path / "idx",
                expected_fingerprint="same-config",
                expected_vectors_checksum="anything",
            )
        assert loaded.size == 3
        assert "rebuild" in caplog.text.lower()
