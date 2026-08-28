"""Settings, derived values, and path layout."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from medsearch.config import (
    ABSOLUTE_MEMORY_FLOOR_GB,
    GENSIM_DEFAULT_BUCKET,
    REFERENCE_CORPUS_SIZE,
    Paths,
    Settings,
    get_settings,
)


class TestDefaults:
    def test_defaults_are_within_the_laptop_budget(self) -> None:
        settings = Settings()
        # 50_000 x 100 dims x 4 bytes = 20 MB, well under the 150 MB budget.
        predicted_mb = settings.fasttext_bucket * settings.vector_size * 4 / 1024**2
        assert predicted_mb < settings.max_artefact_mb

    def test_limit_is_not_a_setting(self) -> None:
        # Sampling must be an explicit per-run flag, never a persisted default
        # that could silently apply to a production run (ADR-007).
        assert not hasattr(Settings(), "limit")

    def test_gensim_default_bucket_constant_is_documented(self) -> None:
        assert GENSIM_DEFAULT_BUCKET == 2_000_000
        assert GENSIM_DEFAULT_BUCKET * 100 * 4 == 800_000_000


class TestEnvironmentOverrides:
    def test_env_var_overrides_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEDSEARCH_VECTOR_SIZE", "64")
        assert Settings().vector_size == 64

    def test_prefix_is_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VECTOR_SIZE", "64")
        assert Settings().vector_size == 100

    def test_explicit_argument_beats_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MEDSEARCH_VECTOR_SIZE", "64")
        assert Settings(vector_size=32).vector_size == 32


class TestValidation:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("vector_size", 8),
            ("vector_size", 1024),
            ("window", 0),
            ("min_count", 0),
            ("epochs", 0),
            ("top_n", 0),
            ("fasttext_bucket", 100),
            ("fasttext_bucket", 5_000_000),
        ],
    )
    def test_out_of_range_values_are_rejected(self, field: str, value: int) -> None:
        with pytest.raises(ValueError):
            Settings(**{field: value})

    def test_max_n_below_min_n_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="fasttext_max_n"):
            Settings(fasttext_min_n=5, fasttext_max_n=2)

    def test_log_level_is_normalised_to_upper(self) -> None:
        assert Settings(log_level="debug").log_level == "DEBUG"

    def test_unknown_log_level_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="log_level"):
            Settings(log_level="chatty")

    def test_unknown_model_name_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            Settings(default_model="word2vec")


class TestEffectiveWorkers:
    @pytest.mark.parametrize(("cores", "expected"), [(1, 1), (2, 1), (4, 3), (8, 7), (16, 15)])
    def test_auto_leaves_one_core_free(
        self, monkeypatch: pytest.MonkeyPatch, cores: int, expected: int
    ) -> None:
        monkeypatch.setattr("medsearch.config.os.cpu_count", lambda: cores)
        assert Settings(workers=0).effective_workers == expected

    def test_explicit_value_is_honoured(self) -> None:
        assert Settings(workers=2).effective_workers == 2

    def test_unknown_cpu_count_falls_back_safely(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("medsearch.config.os.cpu_count", lambda: None)
        assert Settings(workers=0).effective_workers >= 1


class TestMemoryFloor:
    def test_full_corpus_uses_configured_floor(self) -> None:
        assert Settings(min_free_memory_gb=2.0).memory_floor_gb(None) == 2.0

    def test_scales_linearly_with_limit(self) -> None:
        settings = Settings(min_free_memory_gb=4.0)
        half = settings.memory_floor_gb(REFERENCE_CORPUS_SIZE // 2)
        assert half == pytest.approx(2.0, abs=0.01)

    def test_never_below_absolute_floor(self) -> None:
        assert Settings(min_free_memory_gb=2.0).memory_floor_gb(1) == ABSOLUTE_MEMORY_FLOOR_GB

    def test_never_above_configured_ceiling(self) -> None:
        settings = Settings(min_free_memory_gb=2.0)
        assert settings.memory_floor_gb(REFERENCE_CORPUS_SIZE * 10) == 2.0


class TestPaths:
    def test_paths_derive_from_data_dir(self, tmp_path: Path) -> None:
        paths = Settings(data_dir=tmp_path / "d", model_dir=tmp_path / "m").paths
        assert paths.raw_dir == tmp_path / "d" / "raw"
        assert paths.interim_dir == tmp_path / "d" / "interim"
        assert paths.processed_dir == tmp_path / "d" / "processed"

    def test_corpus_file_uses_configured_filename(self, tmp_path: Path) -> None:
        settings = Settings(data_dir=tmp_path, corpus_filename="other.csv")
        assert settings.paths.corpus_file.name == "other.csv"

    def test_model_and_index_paths_are_namespaced(self, tmp_path: Path) -> None:
        paths = Settings(data_dir=tmp_path, model_dir=tmp_path / "m").paths
        assert paths.model_path("skipgram").name == "skipgram"
        assert paths.index_path("fasttext", "title").name == "fasttext-title"

    def test_index_paths_differ_per_model_and_field(self, tmp_path: Path) -> None:
        paths = Settings(data_dir=tmp_path).paths
        generated = {
            paths.index_path(m, f) for m in ("skipgram", "fasttext") for f in ("abstract", "title")
        }
        assert len(generated) == 4

    def test_ensure_creates_every_directory(self, tmp_path: Path) -> None:
        paths = Settings(data_dir=tmp_path / "d", model_dir=tmp_path / "m").paths
        paths.ensure()
        for directory in (paths.raw_dir, paths.interim_dir, paths.processed_dir, paths.model_dir):
            assert directory.is_dir()

    def test_ensure_is_idempotent(self, tmp_path: Path) -> None:
        paths = Settings(data_dir=tmp_path / "d").paths
        paths.ensure()
        paths.ensure()  # must not raise

    def test_paths_is_frozen(self, tmp_path: Path) -> None:
        paths = Settings(data_dir=tmp_path).paths
        with pytest.raises(FrozenInstanceError):
            paths.data_dir = tmp_path / "other"  # type: ignore[misc]

    def test_from_settings_round_trip(self, tmp_path: Path) -> None:
        settings = Settings(data_dir=tmp_path)
        assert Paths.from_settings(settings) == settings.paths


class TestSingleton:
    def test_get_settings_is_cached(self) -> None:
        get_settings.cache_clear()
        assert get_settings() is get_settings()

    def test_cache_clear_re_reads_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        get_settings.cache_clear()
        monkeypatch.setenv("MEDSEARCH_TOP_N", "7")
        try:
            assert get_settings().top_n == 7
        finally:
            get_settings.cache_clear()
