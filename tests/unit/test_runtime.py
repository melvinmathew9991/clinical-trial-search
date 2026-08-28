"""Thread pinning, memory probes, and the preflight guard.

These are the mechanisms that keep a 4-core / 8 GB laptop responsive during a
training run, so they get real tests rather than smoke tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from medsearch.exceptions import ResourceError
from medsearch.runtime import (
    SystemReport,
    available_memory_gb,
    configure_threads,
    cpu_count,
    current_rss_mb,
    free_disk_gb,
    release_memory,
    require_memory,
    system_report,
    total_memory_gb,
    warn_if_memory_tight,
)

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@pytest.mark.filterwarnings("ignore:configure_threads.*:RuntimeWarning")
class TestConfigureThreads:
    """The pytest process has already imported numpy, so every call here
    triggers the ordering guard. That is the correct behaviour -- it is
    asserted explicitly in :meth:`test_warns_when_called_after_numpy` and
    silenced elsewhere so it does not drown the suite output.
    """

    def test_warns_when_called_after_numpy(self) -> None:
        import numpy  # noqa: F401  -- ensure it is in sys.modules

        with pytest.warns(RuntimeWarning, match="after numpy was imported"):
            configure_threads()

    def test_sets_every_blas_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in THREAD_VARS:
            monkeypatch.delenv(var, raising=False)
        configure_threads()
        import os

        for var in THREAD_VARS:
            assert os.environ[var] == "1"

    def test_does_not_override_an_explicit_user_setting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # setdefault semantics: an operator who deliberately set 4 keeps 4.
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        configure_threads()
        import os

        assert os.environ["OMP_NUM_THREADS"] == "4"

    def test_accepts_a_higher_thread_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in THREAD_VARS:
            monkeypatch.delenv(var, raising=False)
        configure_threads(2)
        import os

        assert os.environ["OMP_NUM_THREADS"] == "2"

    def test_zero_is_clamped_to_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in THREAD_VARS:
            monkeypatch.delenv(var, raising=False)
        configure_threads(0)
        import os

        assert os.environ["OMP_NUM_THREADS"] == "1"


class TestProbes:
    def test_cpu_count_is_at_least_one(self) -> None:
        assert cpu_count() >= 1

    def test_memory_probes_are_positive(self) -> None:
        assert total_memory_gb() > 0
        assert available_memory_gb() > 0

    def test_available_never_exceeds_total(self) -> None:
        assert available_memory_gb() <= total_memory_gb()

    def test_rss_is_positive(self) -> None:
        assert current_rss_mb() > 0

    def test_free_disk_on_existing_path(self, tmp_path: Path) -> None:
        assert free_disk_gb(tmp_path) > 0

    def test_free_disk_walks_up_to_an_existing_parent(self, tmp_path: Path) -> None:
        # The probe must work before `paths.ensure()` has created anything.
        assert free_disk_gb(tmp_path / "not" / "yet" / "created") > 0

    def test_release_memory_is_safe_to_call(self) -> None:
        release_memory()


class TestRequireMemory:
    def test_passes_when_memory_is_sufficient(self) -> None:
        require_memory(0.0, stage="test")

    def test_raises_when_memory_is_short(self) -> None:
        with pytest.raises(ResourceError) as exc_info:
            require_memory(10_000.0, stage="train:fasttext")
        assert "train:fasttext" in str(exc_info.value)

    def test_message_reports_both_numbers(self) -> None:
        with pytest.raises(ResourceError) as exc_info:
            require_memory(10_000.0, stage="test")
        message = str(exc_info.value)
        assert "Available:" in message
        assert "Required:" in message

    def test_full_corpus_message_recommends_limit(self) -> None:
        with pytest.raises(ResourceError) as exc_info:
            require_memory(10_000.0, stage="test", limit=None)
        assert "--limit 2000" in str(exc_info.value)
        assert "full corpus" in str(exc_info.value)

    def test_sampled_message_does_not_recommend_limit_again(self) -> None:
        with pytest.raises(ResourceError) as exc_info:
            require_memory(10_000.0, stage="test", limit=500)
        message = str(exc_info.value)
        assert "already sampled" in message
        assert "run a reduced profile with" not in message


class TestWarnIfMemoryTight:
    def test_returns_none_when_comfortable(self) -> None:
        assert warn_if_memory_tight(0.0, stage="test") is None

    def test_returns_message_when_tight(self) -> None:
        warning = warn_if_memory_tight(10_000.0, stage="test")
        assert warning is not None
        assert "test" in warning

    def test_advice_adapts_to_an_existing_limit(self) -> None:
        full = warn_if_memory_tight(10_000.0, stage="test", limit=None)
        sampled = warn_if_memory_tight(10_000.0, stage="test", limit=1000)
        assert full is not None and sampled is not None
        assert "--limit 2000" in full
        assert "already sampled" in sampled


class TestSystemReport:
    def test_report_is_populated(self, tmp_path: Path) -> None:
        report = system_report(workers=3, disk_probe=tmp_path)
        assert isinstance(report, SystemReport)
        assert report.workers == 3
        assert report.cpu_logical >= 1
        assert report.python_version.count(".") == 2

    def test_render_includes_the_key_numbers(self, tmp_path: Path) -> None:
        rendered = system_report(workers=3, disk_probe=tmp_path).render()
        for label in ("Logical cores", "Training workers", "RAM total", "Disk free"):
            assert label in rendered

    def test_render_notes_the_reserved_core(self, tmp_path: Path) -> None:
        assert "reserved" in system_report(workers=3, disk_probe=tmp_path).render()

    def test_defaults_to_cwd_when_no_probe_given(self) -> None:
        assert system_report(workers=1).disk_free_gb > 0
