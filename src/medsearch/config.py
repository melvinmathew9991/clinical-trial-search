"""Central configuration.

Single source of truth for every tunable in the project. Nothing else in
``medsearch`` reads ``os.environ`` directly (Rules.md section 3).

Precedence: defaults -> ``.env`` -> process environment.

Defaults are tuned for the documented target machine: 4 logical cores and
7.89 GB RAM. See Architecture.md section 9 for the resource budget these
values are chosen to satisfy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ModelName = Literal["skipgram", "fasttext"]
FieldName = Literal["abstract", "title"]
Pooling = Literal["mean", "sif"]

#: gensim's default is 2_000_000. At 100 dims x float32 that is exactly
#: 800_000_000 bytes -- precisely the size of the legacy
#: ``model_Fasttext.bin.wv.vectors_ngrams.npy``. See ADR-001.
GENSIM_DEFAULT_BUCKET = 2_000_000

#: Document count of the full Dimensions COVID-19 corpus. The memory floor is
#: calibrated against this, then scaled down for sampled runs.
REFERENCE_CORPUS_SIZE = 10_666

#: Absolute lower bound for the memory floor. Below this, even a tiny sampled
#: run has no headroom for the interpreter, numpy, and gensim's allocations.
ABSOLUTE_MEMORY_FLOOR_GB = 0.5

#: Indexing peaked at 152 MB against training's 346 MB on the full corpus
#: (Architecture.md section 9), so it is gated at half the training floor.
#: Still ~3x the measured need, but it stops a comfortable build being
#: refused on the training budget.
INDEX_FLOOR_FRACTION = 0.5


class Settings(BaseSettings):
    """Runtime settings, overridable by ``MEDSEARCH_*`` environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="MEDSEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------- paths
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    report_dir: Path = Path("reports")
    corpus_filename: str = "dimension-covid.csv"

    # ------------------------------------------------------------- training
    vector_size: int = Field(default=100, ge=16, le=512)
    window: int = Field(default=5, ge=1, le=20)
    min_count: int = Field(default=2, ge=1)
    epochs: int = Field(default=5, ge=1, le=100)
    seed: int = 42

    fasttext_min_n: int = Field(default=3, ge=1, le=6)
    fasttext_max_n: int = Field(default=5, ge=1, le=8)
    fasttext_bucket: int = Field(default=50_000, ge=1_000, le=GENSIM_DEFAULT_BUCKET)

    # ------------------------------------------------------------- resources
    #: 0 means auto -> ``max(1, cpu_count - 1)``, leaving a core for the OS.
    workers: int = Field(default=0, ge=0, le=64)
    chunk_size: int = Field(default=1_000, ge=100)
    min_free_memory_gb: float = Field(default=2.0, ge=0.0)
    warn_free_memory_gb: float = Field(default=3.0, ge=0.0)
    max_artefact_mb: int = Field(default=150, ge=1)

    # ------------------------------------------------------------- pooling
    #: How a document vector is composed from its word vectors.
    #: mean is the original behaviour; sif applies smooth inverse frequency
    #: weighting plus common-component removal (ADR-011).
    pooling: Pooling = "mean"
    #: SIF smoothing constant. Performance is flat over 1e-3 to 1e-4.
    sif_a: float = Field(default=1e-3, gt=0.0, le=1.0)

    # ------------------------------------------------------------- search
    top_n: int = Field(default=10, ge=1, le=100)
    #: Return the union of the embedding and keyword rankings rather than
    #: the embedding ranking alone. Lifts Recall@10 from 0.648 to 0.955 at
    #: the cost of showing ~18 results instead of 10 (PRD 8.3).
    union_retrieval: bool = True
    #: FastText, not Skip-gram. The two are indistinguishable as standalone
    #: rankers (p = 0.28 / 0.86 / 1.00), but under the union that ships they
    #: are not: FastText reaches Recall@10 0.955 against Skip-gram's 0.927
    #: (+0.028, 95% CI [+0.005, +0.052], p = 0.019, paired over 97 queries),
    #: with MRR@10 no worse (+0.030, p = 0.16). It costs a 29.3 MB artefact
    #: against 10.2 MB -- both far inside the 150 MB cap (PRD 8.4).
    default_model: ModelName = "fasttext"
    default_field: FieldName = "abstract"

    # ------------------------------------------------------------- logging
    log_level: str = "INFO"
    log_json: bool = False

    @field_validator("fasttext_max_n")
    @classmethod
    def _max_n_at_least_min_n(cls, v: int, info: object) -> int:
        # pydantic v2 passes a ValidationInfo; guard defensively for typing.
        data = getattr(info, "data", {})
        min_n = data.get("fasttext_min_n", 3)
        if v < min_n:
            msg = f"fasttext_max_n ({v}) must be >= fasttext_min_n ({min_n})"
            raise ValueError(msg)
        return v

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            msg = f"log_level must be one of {sorted(allowed)}, got {v!r}"
            raise ValueError(msg)
        return upper

    # ------------------------------------------------------------- derived
    @property
    def effective_workers(self) -> int:
        """Worker count actually used for training.

        Resolves ``workers=0`` to ``max(1, cpu_count - 1)`` so at least one
        logical core stays free and the desktop remains responsive during a
        training run (Rules.md section 2).
        """
        if self.workers > 0:
            return self.workers
        return max(1, (os.cpu_count() or 2) - 1)

    def memory_floor_gb(self, limit: int | None = None, stage: str = "train") -> float:
        """Free-RAM floor required before a run of this size may start.

        The floor scales with the number of documents actually being
        processed. A flat floor is wrong in both directions: it blocks a
        2,000-row development run that needs ~400 MB, while being no safer for
        the full corpus.

        This previously shipped as a flat ``min_free_memory_gb``, which meant
        the ``ResourceError`` message recommended ``--limit 2000`` as a
        fallback that the same check then refused. Found in Track 0.

        The floor is also stage-aware: indexing peaked at 152 MB against
        training's 346 MB, so gating it on the training budget refused
        builds that fit comfortably.

        Args:
            limit: Row cap for this run, or ``None`` for the full corpus.
            stage: "train" or "index". Indexing gets a lower floor.

        Returns:
            Required free GB, never below :data:`ABSOLUTE_MEMORY_FLOOR_GB` and
            never above ``min_free_memory_gb``.
        """
        ceiling = self.min_free_memory_gb * (INDEX_FLOOR_FRACTION if stage == "index" else 1.0)
        if limit is None:
            return ceiling
        scaled = ceiling * (limit / REFERENCE_CORPUS_SIZE)
        return max(ABSOLUTE_MEMORY_FLOOR_GB, min(scaled, ceiling))

    @property
    def paths(self) -> Paths:
        """Resolved project paths."""
        return Paths.from_settings(self)


@dataclass(frozen=True, slots=True)
class Paths:
    """Resolved filesystem layout.

    Keeping every path in one object means a directory rename touches one
    place, and tests can point the whole project at a ``tmp_path``.
    """

    data_dir: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    model_dir: Path
    report_dir: Path
    corpus_file: Path

    @classmethod
    def from_settings(cls, settings: Settings) -> Paths:
        data = settings.data_dir
        return cls(
            data_dir=data,
            raw_dir=data / "raw",
            interim_dir=data / "interim",
            processed_dir=data / "processed",
            model_dir=settings.model_dir,
            report_dir=settings.report_dir,
            corpus_file=data / "raw" / settings.corpus_filename,
        )

    def model_path(self, model: ModelName) -> Path:
        """Directory holding one model's artefacts."""
        return self.model_dir / model

    def index_path(self, model: ModelName, field: FieldName, pooling: Pooling = "mean") -> Path:
        """Directory holding one document index.

        The pooling method is part of the path so a mean index and a SIF
        index coexist and can be compared without rebuilding either. The
        mean case keeps the original bare name, so artefacts built before
        SIF existed stay addressable.
        """
        suffix = "" if pooling == "mean" else f"-{pooling}"
        return self.processed_dir / f"{model}-{field}{suffix}"

    def ensure(self) -> None:
        """Create every project directory if missing. Idempotent."""
        for path in (
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.model_dir,
            self.report_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so ``.env`` is parsed once. Call ``get_settings.cache_clear()`` in
    tests that need to re-read the environment.
    """
    return Settings()
