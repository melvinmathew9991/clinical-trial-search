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
from typing import cast

import numpy as np

from medsearch.config import FieldName, ModelName, Pooling, Settings
from medsearch.data.loader import corpus_fingerprint, iter_text, load_corpus
from medsearch.embeddings.base import ModelKind, TrainingParams
from medsearch.embeddings.document import DocumentEmbedder, l2_normalize
from medsearch.embeddings.registry import load_metadata, load_vectors, save_model
from medsearch.embeddings.trainer import train_model
from medsearch.embeddings.weighting import (
    SifWeights,
    principal_component,
    remove_component,
)
from medsearch.exceptions import ArtefactMismatchError, ConfigurationError, StaleIndexError
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
    # Floor scales with the number of documents actually processed, so a
    # sampled run is not blocked by the full-corpus requirement.
    require_memory(settings.memory_floor_gb(limit), stage=f"train:{model}", limit=limit)
    if warning := warn_if_memory_tight(
        settings.warn_free_memory_gb, stage=f"train:{model}", limit=limit
    ):
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
    pooling: Pooling | None = None,
) -> DocumentIndex:
    """Embed every document and persist a normalised search index.

    Unlike the legacy pipeline, the index is stamped with the fingerprint of
    the model that produced it, so pairing an index with the wrong model
    raises instead of returning quietly wrong results.
    """
    require_memory(
        settings.memory_floor_gb(limit, stage="index"),
        stage=f"index:{model}",
        limit=limit,
    )

    mode: Pooling = pooling or settings.pooling
    kind = ModelKind(model)
    paths = settings.paths
    model_dir = paths.model_path(model)
    metadata = load_metadata(model_dir, kind)

    cache, count, fingerprint = run_preprocessing(settings, field, limit=limit)

    with stage(f"build_index[{model}-{field}-{mode}]", logger):
        weights = None
        if mode == "sif":
            if SifWeights.exists(model_dir):
                weights = SifWeights.load(model_dir)
                logger.info("Reusing SIF weights from %s", model_dir.name)
            else:
                weights = SifWeights.from_corpus(cache, a=settings.sif_a)
                weights.save(model_dir)

        vectors = load_vectors(model_dir, kind)
        embedder = DocumentEmbedder(vectors, weights=weights)
        matrix = embedder.embed_corpus(cache, chunk_size=settings.chunk_size, total=count)

        component = None
        if mode == "sif":
            # Step 2 of SIF, the one most implementations omit: strip the
            # direction every document shares. It must happen BEFORE L2
            # normalisation, and the same vector is applied to queries.
            component = principal_component(matrix)
            matrix = remove_component(matrix, component)
            logger.info("Removed common component from %d documents", matrix.shape[0])

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
            model_vectors_checksum=metadata.vectors_checksum,
            pooling=mode,
            principal_component=component,
        )
        index.save(paths.index_path(model, field, mode))

    return index


def load_search_engine(
    settings: Settings,
    model: ModelName,
    field: FieldName,
    *,
    limit: int | None = None,
    pooling: Pooling | None = None,
) -> object:
    """Assemble a ready-to-query :class:`~medsearch.search.engine.SearchEngine`.

    Verifies that the index on disk was produced by the model being loaded.
    """
    from medsearch.search.engine import SearchEngine

    kind = ModelKind(model)
    paths = settings.paths
    model_dir = paths.model_path(model)

    mode: Pooling = pooling or settings.pooling
    metadata = load_metadata(model_dir, kind)
    vectors = load_vectors(model_dir, kind)
    index = DocumentIndex.load(
        paths.index_path(model, field, mode),
        expected_fingerprint=metadata.fingerprint,
        expected_vectors_checksum=metadata.vectors_checksum,
    )
    weights = SifWeights.load(model_dir) if index.pooling == "sif" else None

    # Align the corpus with what the index actually covers. Without this a
    # sampled index (2,000 vectors) pairs silently with the full 10,666-row
    # corpus: results stay correct, but 8,666 documents are unreachable and
    # nothing anywhere says so. Same failure class as the legacy artefact
    # mismatch -- wrong pairing, no signal.
    effective_limit = limit if limit is not None else index.sampled_limit
    corpus = load_corpus(paths.corpus_file, limit=effective_limit)

    if index.size != len(corpus):
        raise ArtefactMismatchError(
            expected=f"corpus of {len(corpus)} documents",
            actual=f"index of {index.size} documents ({index.corpus_fingerprint})",
        )

    # Third mismatch class, found in Sprint 11: the corpus file itself changed
    # since the index was built. Row ids are POSITIONAL, so a stale index still
    # resolves -- to the wrong documents. A result would carry one trial's
    # title beside another trial's relevance score, which is worse than an
    # outright failure because nothing looks broken.
    live_fingerprint = corpus_fingerprint(paths.corpus_file)
    expected_corpus = (
        f"{live_fingerprint}-n{index.sampled_limit}" if index.is_sampled else live_fingerprint
    )
    if index.corpus_fingerprint != expected_corpus:
        stale_size = index.size
        index.close()
        raise StaleIndexError(
            expected=expected_corpus,
            actual=index.corpus_fingerprint,
            documents=stale_size,
        )

    if index.is_sampled:
        logger.warning(
            "SAMPLED INDEX: searching %d of the corpus -- this index was built "
            "with --limit %s and is for development only. Rebuild without "
            "--limit for full coverage.",
            index.size,
            index.sampled_limit,
        )

    return SearchEngine(
        index=index,
        # The query must be embedded exactly as the documents were: same
        # weights, and the engine strips the same principal component.
        embedder=DocumentEmbedder(vectors, weights=weights),
        preprocessor=TextPreprocessor(),
        corpus=corpus,
    )


def resolve_models(selection: str) -> list[ModelName]:
    """Expand ``"all"`` into every supported model name.

    Raises:
        ConfigurationError: ``selection`` is not a known model or ``"all"``.
    """
    if selection == "all":
        return ["skipgram", "fasttext"]
    if selection not in ("skipgram", "fasttext"):
        raise ConfigurationError(
            f"Unknown model {selection!r}. Valid choices: skipgram, fasttext, all"
        )
    return [cast(ModelName, selection)]


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


def load_union_retriever(
    settings: Settings,
    model: ModelName,
    field: FieldName,
    *,
    limit: int | None = None,
) -> object:
    """Assemble a :class:`~medsearch.search.hybrid.UnionRetriever`.

    Lives here rather than in ``search`` because wiring persisted artefacts
    together is orchestration: the search layer must not reach up into
    pipelines, and import-linter enforces that.

    The TF-IDF side is built from the same token cache the embedding side was
    trained on, so both retrievers see identical preprocessing. Comparing them
    under different tokenisation would measure the tokeniser, not the method.
    """
    from medsearch.search.baseline import TfidfBaseline
    from medsearch.search.engine import SearchEngine
    from medsearch.search.hybrid import UnionRetriever

    engine = cast("SearchEngine", load_search_engine(settings, model, field, limit=limit))

    # Reuse the engine's corpus rather than loading a second copy: it costs
    # ~35 MB, and an independent load with the caller's `limit` disagrees with
    # the engine whenever the index is sampled -- pairing a 2,000-vector index
    # with a 10,666-row TF-IDF matrix. Same mismatch class the index
    # fingerprint guards against, one layer up.
    effective_limit = limit if limit is not None else engine.sampled_limit
    cache, _, _ = run_preprocessing(settings, field, limit=effective_limit)
    baseline = TfidfBaseline(cache)
    return UnionRetriever(engine, baseline, engine.corpus)
