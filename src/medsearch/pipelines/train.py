"""End-to-end training and indexing orchestration.

Replaces the legacy ``engine.py``, a flat script that trained models, then
threw them away and reloaded from disk, then wrote Skip-gram vectors into
files named FastText (``engine.py`` lines 36 and 41), transposed twice.

Each stage here is separately invocable so the OS reclaims one stage's memory
before the next begins -- the reason ``make train`` runs training and indexing
as two CLI calls rather than one process (Architecture.md section 9).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from medsearch.config import FieldName, ModelName, Settings
from medsearch.data.loader import corpus_fingerprint, iter_text, load_corpus
from medsearch.embeddings.base import ModelKind, TrainingParams
from medsearch.embeddings.document import DocumentEmbedder, l2_normalize
from medsearch.embeddings.registry import load_metadata, load_vectors, save_model
from medsearch.embeddings.trainer import train_model
from medsearch.logging_conf import get_logger, stage
from medsearch.preprocessing.pipeline import TextPreprocessor, TokenCache, preprocess_corpus
from medsearch.runtime import release_memory, require_memory, warn_if_memory_tight
from medsearch.search.index import DocumentIndex

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    """What a training run produced."""

    model: str
    field: str
    documents: int
    vocabulary: int
    artefact_mb: float
    seconds: float
    sampled: bool


def params_from_settings(settings: Settings) -> TrainingParams:
    """Build :class:`TrainingParams` from configuration."""
    return TrainingParams(
        vector_size=settings.vector_size,
        window=settings.window,
        min_count=settings.min_count,
        epochs=settings.epochs,
        workers=settings.effective_workers,
        seed=settings.seed,
        min_n=settings.fasttext_min_n,
        max_n=settings.fasttext_max_n,
        bucket=settings.fasttext_bucket,
    )


def run_preprocessing(
    settings: Settings,
    field: FieldName,
    *,
    limit: int | None = None,
    force: bool = False,
) -> tuple[TokenCache, int, str]:
    """Load the corpus and produce a re-iterable token cache.

    Returns:
        ``(cache, document_count, corpus_fingerprint)``.
    """
    paths = settings.paths
    paths.ensure()

    fingerprint = corpus_fingerprint(paths.corpus_file)
    if limit is not None:
        fingerprint = f"{fingerprint}-n{limit}"

    with stage(f"load_corpus[{field}]", logger):
        frame = load_corpus(paths.corpus_file, limit=limit)
        texts = iter_text(frame, field)
        count = len(texts)

    # Release the frame before preprocessing: it is not needed again until
    # indexing, and holding it costs ~90 MB through the whole token pass.
    del frame
    release_memory()

    cache = TokenCache(paths.interim_dir, fingerprint, field)
    with stage(f"preprocess[{field}]", logger):
        preprocess_corpus(texts, cache, force=force)

    del texts
    release_memory()
    return cache, count, fingerprint


def train_one(
    settings: Settings,
    model: ModelName,
    field: FieldName,
    *,
    limit: int | None = None,
    force: bool = False,
) -> TrainingOutcome:
    """Train a single model and persist it.

    Args:
        settings: Configuration.
        model: ``"skipgram"`` or ``"fasttext"``.
        field: Text field to train on.
        limit: Optional row cap for a development run.
        force: Rebuild the token cache even if present.

    Returns:
        A :class:`TrainingOutcome` summary.
    """
    require_memory(settings.min_free_memory_gb, stage=f"train:{model}")
    if warning := warn_if_memory_tight(settings.warn_free_memory_gb, stage=f"train:{model}"):
        logger.warning(warning)

    kind = ModelKind(model)
    params = params_from_settings(settings)
    cache, count, fingerprint = run_preprocessing(settings, field, limit=limit, force=force)

    vectors, metadata = train_model(
        cache,
        kind=kind,
        params=params,
        corpus_fingerprint=fingerprint,
        document_count=count,
        sampled=limit is not None,
    )

    stamped = save_model(
        vectors,
        metadata,
        settings.paths.model_path(model),
        max_artefact_mb=settings.max_artefact_mb,
    )

    del vectors
    release_memory()

    return TrainingOutcome(
        model=model,
        field=field,
        documents=count,
        vocabulary=stamped.vocabulary_size,
        artefact_mb=round(stamped.artefact_mb, 1),
        seconds=stamped.training_seconds,
        sampled=stamped.sampled,
    )


def build_index(
    settings: Settings,
    model: ModelName,
    field: FieldName,
    *,
    limit: int | None = None,
) -> DocumentIndex:
    """Embed every document and persist a normalised search index.

    Unlike the legacy pipeline, the index is stamped with the fingerprint of
    the model that produced it, so pairing an index with the wrong model
    raises instead of returning quietly wrong results.
    """
    require_memory(settings.min_free_memory_gb, stage=f"index:{model}")

    kind = ModelKind(model)
    paths = settings.paths
    model_dir = paths.model_path(model)
    metadata = load_metadata(model_dir, kind)

    cache, count, fingerprint = run_preprocessing(settings, field, limit=limit)

    with stage(f"build_index[{model}-{field}]", logger):
        vectors = load_vectors(model_dir, kind)
        embedder = DocumentEmbedder(vectors)
        matrix = embedder.embed_corpus(cache, chunk_size=settings.chunk_size, total=count)
        normalised = l2_normalize(matrix)

        del matrix, vectors, embedder
        release_memory()

        index = DocumentIndex(
            vectors=normalised,
            row_ids=np.arange(normalised.shape[0], dtype=np.int64),
            model_fingerprint=metadata.fingerprint,
            model_kind=model,
            field=field,
            corpus_fingerprint=fingerprint,
        )
        index.save(paths.index_path(model, field))

    return index


def load_search_engine(
    settings: Settings,
    model: ModelName,
    field: FieldName,
    *,
    limit: int | None = None,
) -> object:
    """Assemble a ready-to-query :class:`~medsearch.search.engine.SearchEngine`.

    Verifies that the index on disk was produced by the model being loaded.
    """
    from medsearch.search.engine import SearchEngine

    kind = ModelKind(model)
    paths = settings.paths
    model_dir = paths.model_path(model)

    metadata = load_metadata(model_dir, kind)
    vectors = load_vectors(model_dir, kind)
    index = DocumentIndex.load(
        paths.index_path(model, field),
        expected_fingerprint=metadata.fingerprint,
    )
    corpus = load_corpus(paths.corpus_file, limit=limit)

    return SearchEngine(
        index=index,
        embedder=DocumentEmbedder(vectors),
        preprocessor=TextPreprocessor(),
        corpus=corpus,
    )


def resolve_models(selection: str) -> list[ModelName]:
    """Expand ``"all"`` into every supported model name."""
    if selection == "all":
        return ["skipgram", "fasttext"]
    return [selection]  # type: ignore[list-item]


def artefact_report(settings: Settings) -> list[tuple[str, float]]:
    """Sizes of every artefact directory, for ``medsearch doctor``."""
    report: list[tuple[str, float]] = []
    for directory in (settings.model_dir, settings.data_dir):
        if not directory.exists():
            continue
        for child in sorted(directory.rglob("*")):
            if child.is_dir():
                total = sum(f.stat().st_size for f in child.glob("*") if f.is_file())
                if total:
                    report.append((str(child), total / 1024**2))
    return report


def corpus_path(settings: Settings) -> Path:
    """Location of the source corpus."""
    return settings.paths.corpus_file
