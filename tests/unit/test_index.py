"""DocumentIndex persistence, validation, and the sampled-index guard."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from medsearch.exceptions import ArtefactMismatchError, IndexBuildError
from medsearch.search.index import DocumentIndex


def make_index(
    *,
    rows: int = 4,
    dim: int = 4,
    model_fingerprint: str = "fp",
    model_kind: str = "skipgram",
    field: str = "abstract",
    corpus_fingerprint: str = "corpus1",
) -> DocumentIndex:
    """A small valid index."""
    return DocumentIndex(
        vectors=np.eye(rows, dim, dtype=np.float32),
        row_ids=np.arange(rows, dtype=np.int64),
        model_fingerprint=model_fingerprint,
        model_kind=model_kind,
        field=field,
        corpus_fingerprint=corpus_fingerprint,
    )


class TestConstruction:
    def test_size_and_dim(self) -> None:
        index = make_index(rows=7, dim=5)
        assert index.size == 7
        assert index.dim == 5

    def test_nbytes_reflects_float32(self) -> None:
        assert make_index(rows=10, dim=4).nbytes == 10 * 4 * 4

    def test_one_dimensional_vectors_rejected(self) -> None:
        with pytest.raises(IndexBuildError, match="2-D"):
            DocumentIndex(
                vectors=np.zeros(4, dtype=np.float32),
                row_ids=np.arange(4),
                model_fingerprint="f",
                model_kind="skipgram",
                field="abstract",
                corpus_fingerprint="c",
            )

    def test_row_id_length_mismatch_rejected(self) -> None:
        with pytest.raises(IndexBuildError, match="does not match"):
            DocumentIndex(
                vectors=np.eye(4, dtype=np.float32),
                row_ids=np.arange(2),
                model_fingerprint="f",
                model_kind="skipgram",
                field="abstract",
                corpus_fingerprint="c",
            )

    def test_is_frozen(self) -> None:
        index = make_index()
        with pytest.raises(FrozenInstanceError):
            index.model_kind = "fasttext"  # type: ignore[misc]


class TestPersistence:
    def test_round_trip_preserves_vectors(self, tmp_path: Path) -> None:
        original = make_index()
        original.save(tmp_path)
        with DocumentIndex.load(tmp_path) as loaded:
            assert np.allclose(np.asarray(loaded.vectors), np.asarray(original.vectors))

    def test_round_trip_preserves_metadata(self, tmp_path: Path) -> None:
        make_index(model_kind="fasttext", field="title").save(tmp_path)
        with DocumentIndex.load(tmp_path) as loaded:
            assert loaded.model_kind == "fasttext"
            assert loaded.field == "title"

    def test_writes_three_files(self, tmp_path: Path) -> None:
        make_index().save(tmp_path)
        for name in ("vectors.npy", "row_ids.npy", "manifest.json"):
            assert (tmp_path / name).exists()

    def test_vectors_persist_as_float32(self, tmp_path: Path) -> None:
        # ADR-002: binary float32, not a re-parsed CSV.
        make_index().save(tmp_path)
        assert np.load(tmp_path / "vectors.npy").dtype == np.float32

    def test_manifest_is_readable_json(self, tmp_path: Path) -> None:
        make_index().save(tmp_path)
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["dtype"] == "float32"
        assert manifest["normalized"] is True

    def test_creates_missing_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "deeper"
        make_index().save(target)
        assert DocumentIndex.exists(target)

    def test_mmap_load_returns_memmap(self, tmp_path: Path) -> None:
        make_index().save(tmp_path)
        with DocumentIndex.load(tmp_path, mmap=True) as loaded:
            assert isinstance(loaded.vectors, np.memmap)

    def test_non_mmap_load_returns_plain_array(self, tmp_path: Path) -> None:
        make_index().save(tmp_path)
        loaded = DocumentIndex.load(tmp_path, mmap=False)
        assert not isinstance(loaded.vectors, np.memmap)

    def test_missing_index_raises(self, tmp_path: Path) -> None:
        with pytest.raises(IndexBuildError, match="No index"):
            DocumentIndex.load(tmp_path)

    def test_missing_index_message_is_actionable(self, tmp_path: Path) -> None:
        with pytest.raises(IndexBuildError) as exc_info:
            DocumentIndex.load(tmp_path)
        assert "index build" in str(exc_info.value)

    def test_exists_is_false_before_saving(self, tmp_path: Path) -> None:
        assert not DocumentIndex.exists(tmp_path)


class TestFingerprintGuard:
    """The legacy project wrote Skip-gram vectors into FastText files with
    nothing able to detect it. The fingerprint stamp is that detection.
    """

    def test_matching_fingerprint_loads(self, tmp_path: Path) -> None:
        make_index(model_fingerprint="abc").save(tmp_path)
        with DocumentIndex.load(tmp_path, expected_fingerprint="abc") as index:
            assert index.size == 4

    def test_mismatched_fingerprint_raises(self, tmp_path: Path) -> None:
        make_index(model_fingerprint="abc").save(tmp_path)
        with pytest.raises(ArtefactMismatchError, match="fingerprint mismatch"):
            DocumentIndex.load(tmp_path, expected_fingerprint="xyz")

    def test_error_reports_both_fingerprints(self, tmp_path: Path) -> None:
        make_index(model_fingerprint="abc").save(tmp_path)
        with pytest.raises(ArtefactMismatchError) as exc_info:
            DocumentIndex.load(tmp_path, expected_fingerprint="xyz")
        assert "abc" in str(exc_info.value)
        assert "xyz" in str(exc_info.value)

    def test_no_expectation_skips_the_check(self, tmp_path: Path) -> None:
        make_index(model_fingerprint="abc").save(tmp_path)
        with DocumentIndex.load(tmp_path) as index:
            assert index.model_fingerprint == "abc"


class TestSampledIndex:
    @pytest.mark.parametrize(
        ("fingerprint", "expected"),
        [
            ("abc123-n2000", 2000),
            ("abc123-n1", 1),
            ("abc123", None),
            ("", None),
            ("deadbeefncafe", None),
            ("abc-nXYZ", None),
            ("abc-n", None),
        ],
    )
    def test_sampled_limit_parsing(self, fingerprint: str, expected: int | None) -> None:
        assert make_index(corpus_fingerprint=fingerprint).sampled_limit == expected

    def test_is_sampled_flag(self) -> None:
        assert make_index(corpus_fingerprint="abc-n500").is_sampled
        assert not make_index(corpus_fingerprint="abc").is_sampled

    def test_sampled_state_survives_a_round_trip(self, tmp_path: Path) -> None:
        make_index(corpus_fingerprint="abc-n2000").save(tmp_path)
        with DocumentIndex.load(tmp_path) as loaded:
            assert loaded.sampled_limit == 2000


class TestWindowsFileHandle:
    """Windows takes a mandatory lock on a mapped file, so a leaked handle
    blocks `medsearch index build` with WinError 32.
    """

    def test_close_releases_the_mapping(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "idx"
        make_index().save(index_dir)
        loaded = DocumentIndex.load(index_dir, mmap=True)
        loaded.close()
        (index_dir / "vectors.npy").unlink()  # would raise if still held

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        make_index().save(tmp_path)
        loaded = DocumentIndex.load(tmp_path)
        loaded.close()
        loaded.close()

    def test_close_is_safe_without_mmap(self, tmp_path: Path) -> None:
        make_index().save(tmp_path)
        DocumentIndex.load(tmp_path, mmap=False).close()

    def test_context_manager_releases_on_exit(self, tmp_path: Path) -> None:
        index_dir = tmp_path / "idx"
        make_index().save(index_dir)
        with DocumentIndex.load(index_dir) as index:
            assert index.size == 4
        (index_dir / "vectors.npy").unlink()

    def test_index_can_be_overwritten_after_close(self, tmp_path: Path) -> None:
        make_index().save(tmp_path)
        with DocumentIndex.load(tmp_path):
            pass
        make_index(rows=6).save(tmp_path)
        with DocumentIndex.load(tmp_path) as reloaded:
            assert reloaded.size == 6
