"""Regression tests for the four defects found in the legacy Part_1 / Part_2 code.

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
from medsearch.embeddings.base import TrainingParams
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
        assert len(frame) == 5, "the full corpus must be returned when limit is None"

    def test_limit_is_opt_in_and_respected(self, corpus_csv: Path) -> None:
        frame = load_corpus(corpus_csv, limit=2)
        assert len(frame) == 2

    def test_limit_emits_a_warning(self, corpus_csv: Path, caplog: pytest.LogCaptureFixture) -> None:
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
