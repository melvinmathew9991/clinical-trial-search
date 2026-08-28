"""Union retrieval.

Union is the shipped default because it lifts Recall@10 from 0.648 to 0.955
(PRD §8.3). These tests pin the properties that make it worth that: it returns
everything either retriever found, it never loses a document, and consensus
documents rank first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from medsearch.embeddings.document import DocumentEmbedder, l2_normalize
from medsearch.search.baseline import TfidfBaseline
from medsearch.search.engine import SearchEngine
from medsearch.search.hybrid import RRF_K, UnionHit, UnionRetriever
from medsearch.search.index import DocumentIndex


@pytest.fixture
def corpus_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial_id": ["NCT001", "NCT002", "NCT003", "NCT004"],
            "title": ["Lung failure", "Breathing study", "Vaccine trial", "Kidney injury"],
            "abstract": ["a", "b", "c", "d"],
            "publication_date": ["2021-01-01"] * 4,
        }
    )


@pytest.fixture
def retriever(
    fake_vectors: object, stub_preprocessor: object, corpus_frame: pd.DataFrame
) -> UnionRetriever:
    """A retriever whose two halves deliberately disagree.

    The embedding side clusters respiratory terms; the TF-IDF side only matches
    literal tokens. That mirrors the real corpus behaviour the union exists to
    exploit.
    """
    embedder = DocumentEmbedder(fake_vectors)
    documents = [
        ["lung", "failure", "respiratory"],
        ["breathing", "lung"],
        ["vaccine", "antibody"],
        ["kidney", "renal"],
    ]
    index = DocumentIndex(
        vectors=l2_normalize(embedder.embed_corpus(documents)),
        row_ids=np.arange(4, dtype=np.int64),
        model_fingerprint="fp",
        model_kind="skipgram",
        field="abstract",
        corpus_fingerprint="corpus",
    )
    engine = SearchEngine(index, embedder, stub_preprocessor, corpus_frame)
    return UnionRetriever(engine, TfidfBaseline(documents), corpus_frame)


class TestUnion:
    def test_returns_at_most_two_per_method(self, retriever: UnionRetriever) -> None:
        assert len(retriever.search("lung", per_method=2).results) <= 4

    def test_never_returns_fewer_than_either_alone(self, retriever: UnionRetriever) -> None:
        union = {r.trial_id for r in retriever.search("lung failure", per_method=3).results}
        keyword = {
            h.trial_id
            for h in retriever.provenance("lung failure", per_method=3)
            if "keyword" in h.found_by
        }
        semantic = {
            h.trial_id
            for h in retriever.provenance("lung failure", per_method=3)
            if "embedding" in h.found_by
        }
        assert keyword <= union
        assert semantic <= union

    def test_no_duplicate_documents(self, retriever: UnionRetriever) -> None:
        ids = [r.trial_id for r in retriever.search("lung failure", per_method=4).results]
        assert len(ids) == len(set(ids))

    def test_ranks_are_sequential(self, retriever: UnionRetriever) -> None:
        results = retriever.search("lung failure", per_method=4).results
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_scores_descend(self, retriever: UnionRetriever) -> None:
        scores = [r.score for r in retriever.search("lung", per_method=4).results]
        assert scores == sorted(scores, reverse=True)

    def test_consensus_documents_rank_first(self, retriever: UnionRetriever) -> None:
        # RRF sums contributions, so a document both methods return must
        # outrank one only a single method returned.
        hits = retriever.provenance("lung failure", per_method=4)
        consensus = [i for i, h in enumerate(hits) if h.is_consensus]
        single = [i for i, h in enumerate(hits) if not h.is_consensus]
        if consensus and single:
            assert max(consensus) < min(single)

    def test_provenance_records_both_sources(self, retriever: UnionRetriever) -> None:
        sources = {s for h in retriever.provenance("lung", per_method=4) for s in h.found_by}
        assert sources <= {"embedding", "keyword"}

    def test_size_reports_indexed_documents(self, retriever: UnionRetriever) -> None:
        assert retriever.size == 4

    def test_empty_query_degrades_gracefully(self, retriever: UnionRetriever) -> None:
        response = retriever.search("", per_method=4)
        assert response.is_empty
        assert response.reason

    def test_all_oov_query_degrades_gracefully(self, retriever: UnionRetriever) -> None:
        response = retriever.search("zzzz qqqq", per_method=4)
        assert response.is_empty
        assert response.reason is not None
        assert "keyword" in response.reason.lower()

    def test_results_carry_display_fields(self, retriever: UnionRetriever) -> None:
        top = retriever.search("kidney", per_method=4).results[0]
        assert top.trial_id.startswith("NCT")
        assert top.title
        assert top.publication_date

    def test_no_score_is_nan(self, retriever: UnionRetriever) -> None:
        for query in ("lung", "kidney renal", "zzzz", ""):
            for result in retriever.search(query, per_method=4).results:
                assert not np.isnan(result.score)


class TestUnionHit:
    def test_consensus_requires_both_sources(self) -> None:
        assert UnionHit("a", 1.0, ("embedding", "keyword")).is_consensus
        assert not UnionHit("a", 1.0, ("keyword",)).is_consensus

    def test_rrf_constant_is_the_published_default(self) -> None:
        assert RRF_K == 60
