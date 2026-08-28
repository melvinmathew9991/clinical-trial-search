"""Model training.

Replaces the legacy ``train_model.py``, which hardcoded ``workers=5`` on a
4-core machine and left FastText's ``bucket`` at gensim's 2,000,000 default --
the two decisions that made the original project unusable on this laptop.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

from medsearch.embeddings.base import ModelKind, ModelMetadata, TrainingParams
from medsearch.exceptions import ModelError, ResourceError
from medsearch.logging_conf import get_logger, stage
from medsearch.runtime import available_memory_gb, release_memory

logger = get_logger(__name__)

#: Refuse to allocate an n-gram matrix larger than this without an explicit
#: override. 200 MB is already well past the 150 MB artefact budget.
_NGRAM_MATRIX_HARD_LIMIT_BYTES = 200 * 1024**2

#: Rough multiplier for peak training RSS over the n-gram matrix size.
#: gensim holds the input matrix, the output layer, and a copy during
#: normalisation, so budget ~3x plus the word-vector matrix.
_PEAK_MEMORY_FACTOR = 3.0


def _model_fingerprint(kind: ModelKind, params: TrainingParams, corpus_fp: str) -> str:
    """Deterministic id for a (model, hyperparameters, corpus) combination.

    Stamped into both the model metadata and the index manifest, so a mismatch
    can be detected instead of silently returning wrong results.
    """
    payload = "|".join(
        str(part)
        for part in (
            kind.value,
            corpus_fp,
            params.vector_size,
            params.window,
            params.min_count,
            params.epochs,
            params.seed,
            params.min_n,
            params.max_n,
            params.bucket,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _check_memory_budget(kind: ModelKind, params: TrainingParams) -> None:
    """Fail before allocating, not during.

    Raises:
        ResourceError: If the predicted n-gram matrix exceeds the hard limit,
            or if free RAM cannot cover the predicted training peak.
    """
    if kind is not ModelKind.FASTTEXT:
        return

    matrix_bytes = params.ngram_matrix_bytes()
    if matrix_bytes > _NGRAM_MATRIX_HARD_LIMIT_BYTES:
        raise ResourceError(
            f"FastText bucket={params.bucket:,} with vector_size={params.vector_size} "
            f"would allocate a {matrix_bytes / 1024**2:.0f} MB n-gram matrix.\n"
            f"  Limit: {_NGRAM_MATRIX_HARD_LIMIT_BYTES / 1024**2:.0f} MB\n"
            f"  Fix: lower MEDSEARCH_FASTTEXT_BUCKET (default 50000). "
            f"gensim's own default of 2,000,000 produces an 800 MB artefact -- "
            f"see ADR-001 in Architecture.md."
        )

    predicted_gb = (matrix_bytes * _PEAK_MEMORY_FACTOR) / 1024**3
    available = available_memory_gb()
    if available < predicted_gb + 0.5:
        raise ResourceError(
            f"Predicted FastText training peak is ~{predicted_gb:.2f} GB but only "
            f"{available:.2f} GB is available.\n"
            f"  Fix: close other applications, lower MEDSEARCH_FASTTEXT_BUCKET, "
            f"or train on a sample with `--limit 2000`."
        )


def train_model(
    corpus: Iterable[Sequence[str]],
    *,
    kind: ModelKind,
    params: TrainingParams,
    corpus_fingerprint: str,
    document_count: int,
    sampled: bool = False,
) -> tuple[object, ModelMetadata]:
    """Train one embedding model and return it with its provenance.

    Args:
        corpus: Re-iterable of token lists. Pass a
            :class:`~medsearch.preprocessing.pipeline.TokenCache`, not a list
            -- gensim iterates once per epoch and a list holds the whole
            corpus in RAM (ADR-005).
        kind: Skip-gram or FastText.
        params: Hyperparameters. ``params.bucket`` is checked against the
            memory budget before any allocation happens.
        corpus_fingerprint: Digest of the source corpus, for the metadata.
        document_count: Number of documents, for the metadata.
        sampled: True when ``--limit`` was used, recorded so a development
            artefact can never be mistaken for a production one.

    Returns:
        ``(keyed_vectors, metadata)``. Only the ``KeyedVectors`` is returned,
        not the full model -- trainable state is not needed for serving and
        doubles both artefact size and RSS (ADR-006).

    Raises:
        ResourceError: The run would not fit in available memory.
        ModelError: gensim produced an empty vocabulary.
    """
    _check_memory_budget(kind, params)

    from gensim import __version__ as gensim_version
    from gensim.models import FastText, Word2Vec

    kwargs = params.as_gensim_kwargs(kind)
    logger.info(
        "Training %s: vector_size=%d window=%d epochs=%d workers=%d%s",
        kind,
        params.vector_size,
        params.window,
        params.epochs,
        params.workers,
        f" bucket={params.bucket:,}" if kind is ModelKind.FASTTEXT else "",
    )

    started = time.perf_counter()
    with stage(f"train_{kind}", logger):
        if kind is ModelKind.SKIPGRAM:
            model = Word2Vec(corpus_iterable=corpus, **kwargs)
        else:
            model = FastText(corpus_iterable=corpus, **kwargs)
    elapsed = time.perf_counter() - started

    vectors = model.wv
    if len(vectors) == 0:
        raise ModelError(
            f"Training {kind} produced an empty vocabulary.\n"
            f"  Likely cause: min_count={params.min_count} is higher than the "
            f"frequency of every token, or the corpus preprocessed to nothing.\n"
            f"  Fix: lower MEDSEARCH_MIN_COUNT or inspect the token cache."
        )

    # Drop the full model, keeping only the vectors. On FastText this releases
    # the negative-sampling output matrix immediately rather than at the next
    # collection -- worth ~200 MB at the peak.
    del model
    release_memory()

    metadata = ModelMetadata(
        kind=kind.value,
        fingerprint=_model_fingerprint(kind, params, corpus_fingerprint),
        corpus_fingerprint=corpus_fingerprint,
        corpus_documents=document_count,
        vocabulary_size=len(vectors),
        params={
            "vector_size": params.vector_size,
            "window": params.window,
            "min_count": params.min_count,
            "epochs": params.epochs,
            "workers": params.workers,
            "seed": params.seed,
            "min_n": params.min_n,
            "max_n": params.max_n,
            "bucket": params.bucket,
        },
        gensim_version=str(gensim_version),
        artefact_bytes=0,  # filled in by the registry once written
        training_seconds=round(elapsed, 2),
        sampled=sampled,
    )

    logger.info(
        "Trained %s in %.1fs | vocabulary=%d words",
        kind,
        elapsed,
        len(vectors),
    )
    return vectors, metadata


def directory_size_bytes(path: Path) -> int:
    """Total bytes of every file under ``path``."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
