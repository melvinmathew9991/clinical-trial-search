"""Embedding value objects: model kinds, hyperparameters, and provenance."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class ModelKind(str, Enum):
    """Supported embedding architectures.

    Both are trained skip-gram (``sg=1``); FastText adds character n-grams,
    which is what lets it produce a vector for a morphological variant it
    never saw during training.
    """

    SKIPGRAM = "skipgram"
    FASTTEXT = "fasttext"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class TrainingParams:
    """Hyperparameters for one training run.

    Defaults come from :class:`~medsearch.config.Settings` and are tuned for
    a 4-core / 8 GB machine.

    Attributes:
        vector_size: Embedding dimensionality.
        window: Context window radius.
        min_count: Minimum corpus frequency for a word to enter the vocabulary.
        epochs: Passes over the corpus.
        workers: Training threads. Keep at ``cores - 1`` so one logical core
            stays free for the desktop (Rules.md section 2).
        seed: Reproducibility seed. Identical output requires ``workers=1``;
            with more workers, thread interleaving makes runs near-identical
            but not bit-exact.
        min_n: Shortest character n-gram (FastText only).
        max_n: Longest character n-gram (FastText only).
        bucket: Number of hash buckets for character n-grams (FastText only).

            **This is the memory lever of the whole project.** gensim's default
            is 2,000,000; at 100 dims x float32 that is exactly 800,000,000
            bytes -- the size of the legacy
            ``model_Fasttext.bin.wv.vectors_ngrams.npy``. The default here is
            50,000, giving a 20 MB matrix. See ADR-001.
    """

    vector_size: int = 100
    window: int = 5
    min_count: int = 2
    epochs: int = 5
    workers: int = 3
    seed: int = 42
    min_n: int = 3
    max_n: int = 5
    bucket: int = 50_000

    def ngram_matrix_bytes(self) -> int:
        """Predicted size of the FastText n-gram matrix, in bytes.

        Lets ``medsearch doctor`` and the trainer warn *before* allocating
        rather than after the machine starts swapping.
        """
        return self.bucket * self.vector_size * 4  # float32

    def as_gensim_kwargs(self, kind: ModelKind) -> dict[str, Any]:
        """Translate to the keyword arguments gensim expects."""
        common: dict[str, Any] = {
            "vector_size": self.vector_size,
            "window": self.window,
            "min_count": self.min_count,
            "epochs": self.epochs,
            "workers": self.workers,
            "seed": self.seed,
            "sg": 1,
        }
        if kind is ModelKind.FASTTEXT:
            common.update(min_n=self.min_n, max_n=self.max_n, bucket=self.bucket)
        return common


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    """Provenance sidecar written next to every trained artefact.

    Answers, months later, "what produced this file?" -- the question the
    legacy project could not answer, which is how Skip-gram vectors ended up
    inside files named FastText.
    """

    kind: str
    fingerprint: str
    corpus_fingerprint: str
    corpus_documents: int
    vocabulary_size: int
    params: dict[str, Any]
    gensim_version: str
    artefact_bytes: int
    training_seconds: float
    trained_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    sampled: bool = False

    def save(self, path: Path) -> None:
        """Write as ``metadata.json``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> ModelMetadata:
        """Read back from ``metadata.json``."""
        return cls(**json.loads(path.read_text(encoding="utf-8")))

    @property
    def artefact_mb(self) -> float:
        """Artefact size in MB."""
        return self.artefact_bytes / (1024**2)
