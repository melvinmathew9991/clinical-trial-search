"""End-to-end pipeline: corpus -> tokens -> model -> index -> search.

This is the only test that trains a real gensim model. It stays inside the
suite's time budget by using the 20-row fixture corpus with 16 dimensions, one
epoch, one worker and a 1,000 bucket.

It is the test that would have caught the ``corpus_iterable=`` constructor bug
found in Track 0, which no amount of unit testing or type checking did.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from medsearch.config import Settings
from medsearch.embeddings.base import ModelKind
from medsearch.embeddings.registry import is_trained, load_metadata
from medsearch.exceptions import ArtefactMismatchError
from medsearch.pipelines.train import (
    build_index,
    load_search_engine,
    resolve_models,
    run_preprocessing,
    train_one,
)
from medsearch.search.index import DocumentIndex

# Marked slow as well as integration: each case reloads NLTK resources and
# trains a real model (~1.2 s), so the module runs ~40 s. The default `pytest`
# invocation excludes it to keep the developer loop under 30 s (Rules.md
# section 5); `make test-all` and CI run it, and that is where the coverage
# gate is enforced.
pytestmark = [pytest.mark.integration, pytest.mark.slow]


class TestPreprocessingStage:
    def test_produces_a_populated_cache(self, settings: Settings) -> None:
        cache, count, fingerprint = run_preprocessing(settings, "abstract")
        assert count == 20
        assert len(list(cache)) == 20
        assert fingerprint

    def test_second_call_reuses_the_cache(self, settings: Settings) -> None:
        first_cache, _, _ = run_preprocessing(settings, "abstract")
        mtime = first_cache.path.stat().st_mtime_ns
        second_cache, _, _ = run_preprocessing(settings, "abstract")
        assert second_cache.path.stat().st_mtime_ns == mtime

    def test_sampled_run_gets_a_distinct_fingerprint(self, settings: Settings) -> None:
        _, _, full = run_preprocessing(settings, "abstract")
        _, _, sampled = run_preprocessing(settings, "abstract", limit=5)
        assert sampled != full
        assert sampled.endswith("-n5")

    def test_title_and_abstract_caches_are_separate(self, settings: Settings) -> None:
        abstract_cache, _, _ = run_preprocessing(settings, "abstract")
        title_cache, _, _ = run_preprocessing(settings, "title")
        assert abstract_cache.path != title_cache.path


@pytest.mark.parametrize("model", ["skipgram", "fasttext"])
class TestTrainingStage:
    def test_produces_a_loadable_artefact(self, settings: Settings, model: str) -> None:
        outcome = train_one(settings, model, "abstract")  # type: ignore[arg-type]
        assert outcome.documents == 20
        assert outcome.vocabulary > 0
        assert is_trained(settings.paths.model_path(model))  # type: ignore[arg-type]

    def test_metadata_records_provenance(self, settings: Settings, model: str) -> None:
        train_one(settings, model, "abstract")  # type: ignore[arg-type]
        meta = load_metadata(settings.paths.model_path(model), ModelKind(model))  # type: ignore[arg-type]
        assert meta.kind == model
        assert meta.corpus_documents == 20
        assert meta.gensim_version
        assert meta.artefact_bytes > 0
        assert meta.training_seconds >= 0

    def test_full_run_is_not_flagged_as_sampled(self, settings: Settings, model: str) -> None:
        assert train_one(settings, model, "abstract").sampled is False  # type: ignore[arg-type]

    def test_sampled_run_is_flagged(self, settings: Settings, model: str) -> None:
        outcome = train_one(settings, model, "abstract", limit=10)  # type: ignore[arg-type]
        assert outcome.sampled is True
        assert outcome.documents == 10

    def test_artefact_stays_within_budget(self, settings: Settings, model: str) -> None:
        outcome = train_one(settings, model, "abstract")  # type: ignore[arg-type]
        assert outcome.artefact_mb < settings.max_artefact_mb


@pytest.mark.parametrize("model", ["skipgram", "fasttext"])
class TestIndexingStage:
    def test_index_matches_the_corpus(self, settings: Settings, model: str) -> None:
        train_one(settings, model, "abstract")  # type: ignore[arg-type]
        index = build_index(settings, model, "abstract")  # type: ignore[arg-type]
        assert index.size == 20
        assert index.dim == settings.vector_size

    def test_index_rows_are_unit_length_or_zero(self, settings: Settings, model: str) -> None:
        train_one(settings, model, "abstract")  # type: ignore[arg-type]
        index = build_index(settings, model, "abstract")  # type: ignore[arg-type]
        norms = np.linalg.norm(np.asarray(index.vectors), axis=1)
        assert np.all((np.isclose(norms, 1.0)) | (np.isclose(norms, 0.0)))

    def test_index_carries_the_model_fingerprint(self, settings: Settings, model: str) -> None:
        train_one(settings, model, "abstract")  # type: ignore[arg-type]
        build_index(settings, model, "abstract")  # type: ignore[arg-type]
        meta = load_metadata(settings.paths.model_path(model), ModelKind(model))  # type: ignore[arg-type]
        manifest = json.loads(
            (settings.paths.index_path(model, "abstract") / "manifest.json").read_text(  # type: ignore[arg-type]
                encoding="utf-8"
            )
        )
        assert manifest["model_fingerprint"] == meta.fingerprint

    def test_index_is_float32_on_disk(self, settings: Settings, model: str) -> None:
        train_one(settings, model, "abstract")  # type: ignore[arg-type]
        build_index(settings, model, "abstract")  # type: ignore[arg-type]
        path = settings.paths.index_path(model, "abstract") / "vectors.npy"  # type: ignore[arg-type]
        assert np.load(path).dtype == np.float32


class TestSearchStage:
    def _prepare(self, settings: Settings, model: str = "skipgram") -> None:
        train_one(settings, model, "abstract")  # type: ignore[arg-type]
        build_index(settings, model, "abstract")  # type: ignore[arg-type]

    def test_engine_loads_and_searches(self, settings: Settings) -> None:
        self._prepare(settings)
        engine = load_search_engine(settings, "skipgram", "abstract")
        response = engine.search("respiratory failure", top_n=5)  # type: ignore[attr-defined]
        assert not response.is_empty
        assert len(response.results) <= 5

    def test_results_are_ordered_and_finite(self, settings: Settings) -> None:
        self._prepare(settings)
        engine = load_search_engine(settings, "skipgram", "abstract")
        results = engine.search("kidney dialysis", top_n=5).results  # type: ignore[attr-defined]
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert all(np.isfinite(s) for s in scores)

    def test_results_carry_display_fields(self, settings: Settings) -> None:
        self._prepare(settings)
        engine = load_search_engine(settings, "skipgram", "abstract")
        top = engine.search("vaccine antibody", top_n=1).results[0]  # type: ignore[attr-defined]
        assert top.trial_id.startswith("NCT")
        assert top.title
        assert top.publication_date

    def test_gibberish_query_degrades_gracefully(self, settings: Settings) -> None:
        self._prepare(settings)
        engine = load_search_engine(settings, "skipgram", "abstract")
        response = engine.search("zzzzqqqq wibblefrotz", top_n=5)  # type: ignore[attr-defined]
        assert response.is_empty
        assert response.reason

    def test_engine_reports_a_full_index_as_unsampled(self, settings: Settings) -> None:
        self._prepare(settings)
        engine = load_search_engine(settings, "skipgram", "abstract")
        assert engine.is_sampled is False  # type: ignore[attr-defined]

    def test_sampled_index_aligns_the_corpus(self, settings: Settings) -> None:
        train_one(settings, "skipgram", "abstract", limit=10)
        build_index(settings, "skipgram", "abstract", limit=10)
        engine = load_search_engine(settings, "skipgram", "abstract")
        # Corpus is trimmed to the indexed slice, not left at 20 rows.
        assert engine.size == 10  # type: ignore[attr-defined]
        assert engine.is_sampled is True  # type: ignore[attr-defined]


class TestCrossModelGuard:
    def test_index_from_one_model_is_rejected_by_another(self, settings: Settings) -> None:
        """The exact legacy failure: FastText result files holding Skip-gram
        vectors, with nothing able to notice.
        """
        train_one(settings, "skipgram", "abstract")
        train_one(settings, "fasttext", "abstract")
        build_index(settings, "skipgram", "abstract")

        skipgram_index = settings.paths.index_path("skipgram", "abstract")
        fasttext_index = settings.paths.index_path("fasttext", "abstract")
        fasttext_index.mkdir(parents=True, exist_ok=True)
        for name in ("vectors.npy", "row_ids.npy", "manifest.json"):
            (fasttext_index / name).write_bytes((skipgram_index / name).read_bytes())

        with pytest.raises(ArtefactMismatchError):
            load_search_engine(settings, "fasttext", "abstract")


class TestResolveModels:
    def test_all_expands(self) -> None:
        assert resolve_models("all") == ["skipgram", "fasttext"]

    def test_single_model_passes_through(self) -> None:
        assert resolve_models("fasttext") == ["fasttext"]

    def test_unknown_model_raises(self) -> None:
        from medsearch.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError, match="Unknown model"):
            resolve_models("word2vec")


class TestReproducibility:
    def test_same_seed_and_single_worker_gives_stable_vectors(self, settings: Settings) -> None:
        train_one(settings, "skipgram", "abstract")
        first = build_index(settings, "skipgram", "abstract")
        first_copy = np.array(first.vectors, copy=True)
        first.close()

        train_one(settings, "skipgram", "abstract", force=True)
        second = build_index(settings, "skipgram", "abstract")
        assert np.allclose(first_copy, np.asarray(second.vectors), atol=1e-5)
        second.close()


class TestIndexLifecycle:
    def test_index_can_be_rebuilt_after_being_loaded(self, settings: Settings) -> None:
        # Windows holds a mandatory lock on a mapped file; a leaked handle
        # would make this raise WinError 32.
        train_one(settings, "skipgram", "abstract")
        build_index(settings, "skipgram", "abstract")
        directory = settings.paths.index_path("skipgram", "abstract")
        with DocumentIndex.load(directory) as index:
            assert index.size == 20
        build_index(settings, "skipgram", "abstract")
