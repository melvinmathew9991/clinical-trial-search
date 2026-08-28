"""Model artefact persistence and provenance metadata."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from medsearch.embeddings.base import ModelKind, ModelMetadata, TrainingParams
from medsearch.embeddings.registry import (
    is_trained,
    load_metadata,
    metadata_path,
    save_model,
    vectors_path,
)
from medsearch.exceptions import ModelNotTrainedError


class SavableVectors:
    """A WordVectors-shaped fake that can persist itself."""

    def __init__(self, size: int = 4, words: int = 10) -> None:
        self.vector_size = size
        self.index_to_key = [f"w{i}" for i in range(words)]
        self.bucket = 0
        self._data = {k: np.ones(size, dtype=np.float32) for k in self.index_to_key}

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: object) -> object:
        return self._data[key]  # type: ignore[index]

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"x" * 2048)


def metadata(**overrides: object) -> ModelMetadata:
    base = {
        "kind": "skipgram",
        "fingerprint": "fp123",
        "corpus_fingerprint": "corpus1",
        "corpus_documents": 20,
        "vocabulary_size": 10,
        "params": {"vector_size": 4},
        "gensim_version": "4.4.0",
        "artefact_bytes": 0,
        "training_seconds": 1.5,
    }
    base.update(overrides)
    return ModelMetadata(**base)  # type: ignore[arg-type]


class TestPaths:
    def test_filenames_are_fixed(self, tmp_path: Path) -> None:
        # Constant inner names remove the FastText-vec vs Fasttext-vec class
        # of bug entirely.
        assert vectors_path(tmp_path).name == "model.kv"
        assert metadata_path(tmp_path).name == "metadata.json"

    def test_filenames_are_lowercase(self, tmp_path: Path) -> None:
        assert vectors_path(tmp_path).name.islower()


class TestSaveModel:
    def test_writes_both_artefacts(self, tmp_path: Path) -> None:
        save_model(SavableVectors(), metadata(), tmp_path)
        assert vectors_path(tmp_path).exists()
        assert metadata_path(tmp_path).exists()

    def test_creates_the_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "models" / "skipgram"
        save_model(SavableVectors(), metadata(), target)
        assert is_trained(target)

    def test_artefact_bytes_is_measured_not_trusted(self, tmp_path: Path) -> None:
        stamped = save_model(SavableVectors(), metadata(artefact_bytes=0), tmp_path)
        assert stamped.artefact_bytes == 2048

    def test_metadata_json_is_valid(self, tmp_path: Path) -> None:
        save_model(SavableVectors(), metadata(), tmp_path)
        loaded = json.loads(metadata_path(tmp_path).read_text(encoding="utf-8"))
        assert loaded["kind"] == "skipgram"
        assert loaded["gensim_version"] == "4.4.0"

    def test_over_budget_artefact_warns(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            save_model(SavableVectors(), metadata(), tmp_path, max_artefact_mb=0)
        assert "over the" in caplog.text
        assert "BUCKET" in caplog.text

    def test_within_budget_does_not_warn(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            save_model(SavableVectors(), metadata(), tmp_path, max_artefact_mb=150)
        assert "over the" not in caplog.text

    def test_sampled_flag_is_preserved(self, tmp_path: Path) -> None:
        stamped = save_model(SavableVectors(), metadata(sampled=True), tmp_path)
        assert stamped.sampled is True


class TestLoad:
    def test_metadata_round_trip(self, tmp_path: Path) -> None:
        save_model(SavableVectors(), metadata(), tmp_path)
        loaded = load_metadata(tmp_path, ModelKind.SKIPGRAM)
        assert loaded.fingerprint == "fp123"
        assert loaded.corpus_documents == 20

    def test_missing_metadata_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ModelNotTrainedError, match="skipgram"):
            load_metadata(tmp_path, ModelKind.SKIPGRAM)

    def test_missing_model_message_is_actionable(self, tmp_path: Path) -> None:
        with pytest.raises(ModelNotTrainedError) as exc_info:
            load_metadata(tmp_path, ModelKind.FASTTEXT)
        assert "medsearch train" in str(exc_info.value)

    def test_is_trained_requires_both_files(self, tmp_path: Path) -> None:
        assert not is_trained(tmp_path)
        vectors_path(tmp_path).write_bytes(b"x")
        assert not is_trained(tmp_path)
        metadata().save(metadata_path(tmp_path))
        assert is_trained(tmp_path)


class TestModelMetadata:
    def test_artefact_mb_conversion(self) -> None:
        assert metadata(artefact_bytes=1024**2).artefact_mb == pytest.approx(1.0)

    def test_trained_at_is_iso_utc(self) -> None:
        assert metadata().trained_at.endswith("+00:00")

    def test_defaults_to_not_sampled(self) -> None:
        assert metadata().sampled is False

    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        original = metadata(vocabulary_size=99)
        target = tmp_path / "metadata.json"
        original.save(target)
        assert ModelMetadata.load(target).vocabulary_size == 99

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "a" / "b" / "metadata.json"
        metadata().save(target)
        assert target.exists()


class TestTrainingParams:
    def test_ngram_matrix_bytes_arithmetic(self) -> None:
        assert TrainingParams(vector_size=100, bucket=50_000).ngram_matrix_bytes() == 20_000_000

    def test_gensim_default_reproduces_the_legacy_800mb(self) -> None:
        assert TrainingParams(vector_size=100, bucket=2_000_000).ngram_matrix_bytes() == 800_000_000

    def test_skipgram_kwargs_omit_ngram_settings(self) -> None:
        kwargs = TrainingParams().as_gensim_kwargs(ModelKind.SKIPGRAM)
        assert "bucket" not in kwargs
        assert "min_n" not in kwargs

    def test_fasttext_kwargs_include_ngram_settings(self) -> None:
        kwargs = TrainingParams().as_gensim_kwargs(ModelKind.FASTTEXT)
        assert kwargs["bucket"] == 50_000
        assert kwargs["min_n"] == 3

    def test_both_models_train_skipgram_style(self) -> None:
        for kind in ModelKind:
            assert TrainingParams().as_gensim_kwargs(kind)["sg"] == 1

    def test_params_are_frozen(self) -> None:
        params = TrainingParams()
        with pytest.raises(FrozenInstanceError):
            params.bucket = 999  # type: ignore[misc]


class TestModelKind:
    def test_string_value(self) -> None:
        assert str(ModelKind.SKIPGRAM) == "skipgram"

    def test_constructible_from_string(self) -> None:
        assert ModelKind("fasttext") is ModelKind.FASTTEXT

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValueError):
            ModelKind("word2vec")
