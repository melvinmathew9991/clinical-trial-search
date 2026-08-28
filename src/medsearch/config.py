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

#: gensim's default is 2_000_000. At 100 dims x float32 that is exactly
#: 800_000_000 bytes -- precisely the size of the legacy
#: ``model_Fasttext.bin.wv.vectors_ngrams.npy``. See ADR-001.
GENSIM_DEFAULT_BUCKET = 2_000_000


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

    # ------------------------------------------------------------- search
    top_n: int = Field(default=10, ge=1, le=100)
    default_model: ModelName = "skipgram"
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

    def index_path(self, model: ModelName, field: FieldName) -> Path:
        """Directory holding one document index."""
        return self.processed_dir / f"{model}-{field}"

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
