"""Runtime and system-resource management.

This module exists because the project's target machine is a 4-logical-core,
7.89 GB laptop, and the legacy implementation made it unresponsive. It is
imported first by every entrypoint.

Three jobs:

1. **Thread pinning** -- stop numpy/BLAS from spawning one thread per core on
   top of gensim's own worker pool. Must run *before* numpy is imported.
2. **Memory probing** -- report free RAM and peak RSS so stage boundaries can
   be logged and budgets enforced.
3. **Preflight** -- refuse to start an expensive stage when the machine cannot
   afford it, rather than triggering a swap storm.

See Architecture.md section 9 (resource budget) and ADR-008.
"""

from __future__ import annotations

import gc
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from medsearch.exceptions import ResourceError

#: Environment variables that control BLAS / OpenMP thread pools. Each of
#: numpy, scipy and gensim may consult a different one depending on which
#: BLAS the wheel was built against, so all are set.
_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

#: NLTK corpora the preprocessing pipeline requires.
_NLTK_PACKAGES = ("stopwords", "wordnet", "punkt", "punkt_tab", "omw-1.4")

_BYTES_PER_GB = 1024**3
_BYTES_PER_MB = 1024**2


def configure_threads(n_threads: int = 1) -> None:
    """Pin BLAS/OpenMP thread pools.

    Without this, a training run on a 4-core machine spawns gensim's workers
    *and* a numpy thread pool sized to the core count. The two oversubscribe
    the CPU, the scheduler thrashes, and the desktop stops responding --
    exactly the lag the legacy project produced with ``workers=5`` on 4 cores.

    Must be called before numpy is first imported; the variables are read at
    library load time. Entrypoints call it at module top level.

    Args:
        n_threads: Threads per BLAS pool. Keep at 1 and let gensim's
            ``workers`` provide the parallelism.
    """
    value = str(max(1, n_threads))
    for var in _THREAD_ENV_VARS:
        os.environ.setdefault(var, value)

    if "numpy" in sys.modules:  # pragma: no cover - ordering guard
        import warnings

        warnings.warn(
            "configure_threads() was called after numpy was imported; "
            "BLAS thread limits will not take effect this process.",
            RuntimeWarning,
            stacklevel=2,
        )


# ---------------------------------------------------------------- probes
def cpu_count() -> int:
    """Logical core count, never less than 1."""
    return os.cpu_count() or 1


def available_memory_gb() -> float:
    """Currently available (not merely free) system RAM, in GB.

    Falls back to :data:`float('inf')` when ``psutil`` is unavailable, so a
    missing optional dependency degrades to "no preflight" rather than a hard
    failure.
    """
    try:
        import psutil
    except ImportError:  # pragma: no cover
        return float("inf")
    return float(psutil.virtual_memory().available) / _BYTES_PER_GB


def total_memory_gb() -> float:
    """Total installed RAM in GB."""
    try:
        import psutil
    except ImportError:  # pragma: no cover
        return float("inf")
    return float(psutil.virtual_memory().total) / _BYTES_PER_GB


def current_rss_mb() -> float:
    """Resident set size of this process, in MB."""
    try:
        import psutil
    except ImportError:  # pragma: no cover
        return 0.0
    return float(psutil.Process().memory_info().rss) / _BYTES_PER_MB


def free_disk_gb(path: Path) -> float:
    """Free space on the volume containing ``path``, in GB."""
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free / _BYTES_PER_GB


def release_memory() -> None:
    """Force a collection at a stage boundary.

    Called after dropping a large intermediate (the raw frame, the token
    cache) so the next stage starts from a clean baseline instead of holding
    two stages' peaks at once.
    """
    gc.collect()


# ---------------------------------------------------------------- preflight
@dataclass(frozen=True, slots=True)
class SystemReport:
    """Snapshot of machine capacity, rendered by ``medsearch doctor``."""

    cpu_logical: int
    workers: int
    memory_total_gb: float
    memory_available_gb: float
    process_rss_mb: float
    disk_free_gb: float
    python_version: str
    thread_limits: dict[str, str]

    def render(self) -> str:
        """Human-readable multi-line summary."""
        limits = ", ".join(f"{k}={v}" for k, v in sorted(self.thread_limits.items()))
        return "\n".join(
            [
                "System report",
                "-" * 52,
                f"  Python            : {self.python_version}",
                f"  Logical cores     : {self.cpu_logical}",
                f"  Training workers  : {self.workers}  (one core reserved for the OS)",
                f"  RAM total         : {self.memory_total_gb:.2f} GB",
                f"  RAM available     : {self.memory_available_gb:.2f} GB",
                f"  This process RSS  : {self.process_rss_mb:.0f} MB",
                f"  Disk free         : {self.disk_free_gb:.1f} GB",
                f"  BLAS thread limits: {limits or 'not set'}",
            ]
        )


def system_report(workers: int, disk_probe: Path | None = None) -> SystemReport:
    """Collect a :class:`SystemReport` for the current machine."""
    return SystemReport(
        cpu_logical=cpu_count(),
        workers=workers,
        memory_total_gb=total_memory_gb(),
        memory_available_gb=available_memory_gb(),
        process_rss_mb=current_rss_mb(),
        disk_free_gb=free_disk_gb(disk_probe or Path.cwd()),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        thread_limits={v: os.environ[v] for v in _THREAD_ENV_VARS if v in os.environ},
    )


def require_memory(minimum_gb: float, *, stage: str) -> None:
    """Raise before an expensive stage if RAM is short.

    Failing here costs a second. Failing by swapping costs the session --
    on an 8 GB machine an over-budget gensim run will page for tens of
    minutes and can take the window manager down with it.

    Args:
        minimum_gb: Floor below which the stage must not start.
        stage: Stage name, for the error message.

    Raises:
        ResourceError: When available memory is below ``minimum_gb``.
    """
    available = available_memory_gb()
    if available < minimum_gb:
        raise ResourceError(
            f"Not enough free memory to start '{stage}'.\n"
            f"  Available: {available:.2f} GB\n"
            f"  Required:  {minimum_gb:.2f} GB\n"
            f"  Fix: close other applications, or run a reduced profile with "
            f"`--limit 2000`, which needs roughly 400 MB."
        )


def warn_if_memory_tight(threshold_gb: float, *, stage: str) -> str | None:
    """Return a warning message when RAM is low but above the hard floor."""
    available = available_memory_gb()
    if available < threshold_gb:
        return (
            f"Only {available:.2f} GB RAM available before '{stage}'. "
            f"Expect slow going; consider `--limit 2000` or closing other apps."
        )
    return None


# ---------------------------------------------------------------- nltk
def ensure_nltk_data(quiet: bool = True) -> None:
    """Download the NLTK corpora the pipeline needs, once.

    The legacy code called ``nltk.download()`` at *module import* in five
    separate files, so every process start hit the network. Here it is an
    explicit, idempotent setup step: already-present packages are skipped.
    """
    import nltk

    for package in _NLTK_PACKAGES:
        try:
            nltk.data.find(f"corpora/{package}")
        except LookupError:
            try:
                nltk.data.find(f"tokenizers/{package}")
            except LookupError:
                nltk.download(package, quiet=quiet)


def _main() -> int:
    """``python -m medsearch.runtime --download-nltk`` (used by ``make setup``)."""
    if "--download-nltk" in sys.argv:
        configure_threads()
        ensure_nltk_data(quiet=False)
        return 0
    configure_threads()
    print(system_report(workers=max(1, cpu_count() - 1)).render())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
