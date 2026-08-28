"""Artefact persistence.

Each model lives in its own directory with fixed inner filenames::

    models/skipgram/
        model.kv            # KeyedVectors -- serving artefact
        model.kv.vectors.npy
        metadata.json       # provenance

Directory-per-model with constant inner names removes the whole class of bug
the legacy project hit, where ``FastText-vec-abstract.csv`` was written but
``Fasttext-vec-abstract.csv`` was read -- a mismatch that worked on Windows
and raised ``FileNotFoundError`` on Linux.
"""

from __future__ import annotations

from pathlib import Path

from medsearch._typing import WordVectors
from medsearch.embeddings.base import ModelKind, ModelMetadata
from medsearch.exceptions import ModelNotTrainedError
from medsearch.logging_conf import get_logger

logger = get_logger(__name__)

_VECTORS_FILENAME = "model.kv"
_METADATA_FILENAME = "metadata.json"


def vectors_path(model_dir: Path) -> Path:
    """Path to the serialised ``KeyedVectors``."""
    return model_dir / _VECTORS_FILENAME


def metadata_path(model_dir: Path) -> Path:
    """Path to the provenance sidecar."""
    return model_dir / _METADATA_FILENAME


def save_model(
    vectors: WordVectors,
    metadata: ModelMetadata,
    model_dir: Path,
    *,
    max_artefact_mb: int = 150,
) -> ModelMetadata:
    """Persist vectors plus metadata, and verify the size budget.

    Args:
        vectors: gensim ``KeyedVectors``.
        metadata: Provenance; ``artefact_bytes`` is filled in here.
        model_dir: Destination directory, created if absent.
        max_artefact_mb: Budget ceiling. Exceeding it logs a warning rather
            than raising -- the artefact is already on disk by then, and the
            operator needs to see the number to act on it.

    Returns:
        The metadata with ``artefact_bytes`` populated.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    target = vectors_path(model_dir)
    vectors.save(str(target))  # type: ignore[attr-defined]  # not on the read protocol

    total_bytes = sum(f.stat().st_size for f in model_dir.glob("model.kv*") if f.is_file())
    stamped = ModelMetadata(
        kind=metadata.kind,
        fingerprint=metadata.fingerprint,
        corpus_fingerprint=metadata.corpus_fingerprint,
        corpus_documents=metadata.corpus_documents,
        vocabulary_size=metadata.vocabulary_size,
        params=metadata.params,
        gensim_version=metadata.gensim_version,
        artefact_bytes=total_bytes,
        training_seconds=metadata.training_seconds,
        trained_at=metadata.trained_at,
        sampled=metadata.sampled,
    )
    stamped.save(metadata_path(model_dir))

    size_mb = total_bytes / (1024**2)
    if size_mb > max_artefact_mb:
        logger.warning(
            "Artefact %s is %.0f MB, over the %d MB budget. "
            "Lower MEDSEARCH_FASTTEXT_BUCKET or vector_size (ADR-001).",
            model_dir.name,
            size_mb,
            max_artefact_mb,
        )
    else:
        logger.info("Saved %s (%.1f MB) to %s", metadata.kind, size_mb, model_dir)

    return stamped


def load_vectors(model_dir: Path, kind: ModelKind, *, mmap: bool = True) -> WordVectors:
    """Load a model's ``KeyedVectors`` for serving.

    Args:
        model_dir: Directory written by :func:`save_model`.
        kind: Which model, used only for the error message.
        mmap: Memory-map the vector array instead of reading it into RSS.
            The array is read-only at serving time, so this is free memory.

    Raises:
        ModelNotTrainedError: No artefact at ``model_dir``.
    """
    target = vectors_path(model_dir)
    if not target.exists():
        raise ModelNotTrainedError(kind.value, target)

    from gensim.models import KeyedVectors

    vectors: WordVectors = KeyedVectors.load(str(target), mmap="r" if mmap else None)
    logger.debug("Loaded %s vectors: %d words", kind, len(vectors))
    return vectors


def load_metadata(model_dir: Path, kind: ModelKind) -> ModelMetadata:
    """Load a model's provenance sidecar.

    Raises:
        ModelNotTrainedError: No ``metadata.json`` at ``model_dir``.
    """
    target = metadata_path(model_dir)
    if not target.exists():
        raise ModelNotTrainedError(kind.value, target)
    return ModelMetadata.load(target)


def is_trained(model_dir: Path) -> bool:
    """True when both the vectors and the metadata are present."""
    return vectors_path(model_dir).exists() and metadata_path(model_dir).exists()
