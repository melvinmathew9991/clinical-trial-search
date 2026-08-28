"""SearchEngine: ranking correctness and graceful degradation.

Index persistence lives in ``test_index.py``; this module is about what the
engine does with a loaded index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from medsearch.embeddings.document import DocumentEmbedder, l2_normalize
from medsearch.search.engine import SearchEngine, SearchResponse
from medsearch.search.index import DocumentIndex


@pytest.fixture
def engine(fake_vectors: object, stub_preprocessor: object) -> SearchEngine:
    """A four-document engine with known semantic structure."""
    embedder = DocumentEmbedder(fake_vectors)
    documents = [
        ["lung", "failure", "respiratory"],  # 0 - respiratory cluster
        ["breathing", "lung"],  # 1 - respiratory cluster
        ["vaccine", "antibody"],  # 2 - immunology cluster
        ["kidney", "renal"],  # 3 - renal cluster
    ]
    matrix = l2_normalize(embedder.embed_corpus(documents))
    index = DocumentIndex(
        vectors=matrix,
        row_ids=np.arange(4, dtype=np.int64),
        model_fingerprint="fp",
        model_kind="skipgram",
        field="abstract",
        corpus_fingerprint="corpus",
    )
    corpus = pd.DataFrame(
        {
            "trial_id": ["NCT001", "NCT002", "NCT003", "NCT004"],
            "title": ["Lung failure", "Breathing study", "Vaccine trial", "Kidney injury"],
            "abstract": ["a" * 400, "b", "c", "d"],
            "publication_date": ["2021-01-01"] * 4,
        }
    )
    return SearchEngine(index, embedder, stub_preprocessor, corpus)


class TestRanking:
    def test_semantically_closest_document_ranks_first(self, engine: SearchEngine) -> None:
        response = engine.search("respiratory failure", top_n=4)
        assert response.results[0].trial_id in {"NCT001", "NCT002"}

    def test_renal_query_beats_respiratory_documents(self, engine: SearchEngine) -> None:
        response = engine.search("renal kidney", top_n=4)
        assert response.results[0].trial_id == "NCT004"

    def test_scores_are_descending(self, engine: SearchEngine) -> None:
        response = engine.search("lung", top_n=4)
        scores = [r.score for r in response.results]
        assert scores == sorted(scores, reverse=True)

    def test_scores_are_valid_cosine_values(self, engine: SearchEngine) -> None:
        response = engine.search("lung breathing", top_n=4)
        for result in response.results:
            assert -1.0001 <= result.score <= 1.0001
            assert not np.isnan(result.score)

    def test_top_n_is_respected(self, engine: SearchEngine) -> None:
        assert len(engine.search("lung", top_n=2).results) <= 2

    def test_top_n_larger_than_corpus_is_safe(self, engine: SearchEngine) -> None:
        response = engine.search("lung", top_n=999)
        assert len(response.results) <= engine.size

    def test_ranks_are_sequential(self, engine: SearchEngine) -> None:
        response = engine.search("lung failure", top_n=3)
        assert [r.rank for r in response.results] == list(range(1, len(response.results) + 1))


class TestGracefulDegradation:
    """An unanswerable query is a first-class outcome, not an exception."""

    def test_empty_query_returns_reason(self, engine: SearchEngine) -> None:
        response = engine.search("", top_n=5)
        assert response.is_empty
        assert response.reason is not None
        assert "empty" in response.reason.lower()

    def test_whitespace_query_returns_reason(self, engine: SearchEngine) -> None:
        assert engine.search("   ", top_n=5).is_empty

    def test_all_oov_query_explains_itself(self, engine: SearchEngine) -> None:
        response = engine.search("zzzz qqqq", top_n=5)
        assert response.is_empty
        assert response.reason is not None
        assert "vocabulary" in response.reason.lower()

    def test_no_result_ever_carries_nan(self, engine: SearchEngine) -> None:
        for query in ("", "zzzz", "lung", "kidney vaccine"):
            for result in engine.search(query, top_n=5).results:
                assert not np.isnan(result.score)


class TestTopIndexSelection:
    """``argpartition`` selection must match a full sort."""

    @pytest.mark.parametrize("top_n", [1, 3, 5, 10])
    def test_matches_argsort(self, top_n: int) -> None:
        rng = np.random.default_rng(7)
        scores = rng.random(50)
        chosen = SearchEngine._top_indices(scores, top_n)
        expected = np.argsort(scores)[::-1][:top_n]
        assert np.array_equal(scores[chosen], scores[expected])

    def test_zero_top_n_returns_empty(self) -> None:
        assert SearchEngine._top_indices(np.array([0.5, 0.2]), 0).size == 0


class TestSearchResponse:
    def test_to_frame_on_empty_response_has_expected_columns(self) -> None:
        frame = SearchResponse(query="x", results=[], reason="none").to_frame()
        assert list(frame.columns) == [
            "rank",
            "score",
            "trial_id",
            "title",
            "abstract",
            "publication_date",
        ]
        assert frame.empty

    def test_abstract_truncation(self, engine: SearchEngine) -> None:
        response = engine.search("lung failure respiratory", top_n=1)
        result = response.results[0]
        if len(result.abstract) > 50:
            truncated = result.truncated_abstract(50)
            assert len(truncated) <= 53
            assert truncated.endswith("...")


class TestEngineContracts:
    def test_rejects_dimension_mismatch_with_the_model(
        self, fake_vectors: object, stub_preprocessor: object
    ) -> None:
        index = DocumentIndex(
            vectors=np.zeros((2, 99), dtype=np.float32),
            row_ids=np.arange(2),
            model_fingerprint="f",
            model_kind="skipgram",
            field="abstract",
            corpus_fingerprint="c",
        )
        with pytest.raises(ValueError, match="does not match"):
            SearchEngine(index, DocumentEmbedder(fake_vectors), stub_preprocessor, pd.DataFrame())

    def test_size_reports_indexed_documents(self, engine: SearchEngine) -> None:
        assert engine.size == 4

    def test_is_sampled_is_false_for_a_full_index(self, engine: SearchEngine) -> None:
        assert engine.is_sampled is False

    def test_is_sampled_is_true_for_a_sampled_index(
        self, fake_vectors: object, stub_preprocessor: object
    ) -> None:
        index = DocumentIndex(
            vectors=np.eye(2, 4, dtype=np.float32),
            row_ids=np.arange(2),
            model_fingerprint="f",
            model_kind="skipgram",
            field="abstract",
            corpus_fingerprint="abc-n2000",
        )
        corpus = pd.DataFrame(
            {
                "trial_id": ["A", "B"],
                "title": ["t", "t"],
                "abstract": ["a", "a"],
                "publication_date": ["2021", "2021"],
            }
        )
        engine = SearchEngine(index, DocumentEmbedder(fake_vectors), stub_preprocessor, corpus)
        assert engine.is_sampled is True
