"""Okapi BM25.

The tests that matter are in :class:`TestRankingProperties` — they pin the two
behaviours that distinguish BM25 from the TF-IDF baseline sitting next to it.
If those stop holding, the module has silently become a slower TF-IDF.
"""

from __future__ import annotations

import numpy as np
import pytest

from medsearch.search.bm25 import DEFAULT_B, DEFAULT_K1, BM25Baseline

DOCS: list[list[str]] = [
    ["lung", "failure"],
    ["vaccine", "efficacy", "trial"],
    ["colchicine", "inflammation"],
    ["remdesivir", "antiviral", "covid19"],
    ["covid19", "vaccine"],
]


@pytest.fixture
def bm25() -> BM25Baseline:
    return BM25Baseline(DOCS)


class TestConstruction:
    def test_indexes_every_document(self, bm25: BM25Baseline) -> None:
        assert bm25.size == len(DOCS)

    def test_matrix_stays_sparse(self, bm25: BM25Baseline) -> None:
        import scipy.sparse as sp

        assert sp.issparse(bm25._matrix)

    def test_matrix_is_float32(self, bm25: BM25Baseline) -> None:
        assert bm25._matrix.dtype == np.float32

    def test_defaults_are_the_trec_values(self) -> None:
        """Not tuned: a baseline tuned on its own eval set is not a baseline."""
        assert (DEFAULT_K1, DEFAULT_B) == (1.2, 0.75)


class TestRankingProperties:
    """What makes this BM25 rather than TF-IDF."""

    def test_term_frequency_saturates(self) -> None:
        """Ten occurrences must not score ten times one occurrence.

        This is the property TF-IDF lacks, and it matters on trial abstracts,
        which repeat the intervention name throughout.
        """
        once = BM25Baseline([["colchicine"], ["other"]])
        many = BM25Baseline([["colchicine"] * 10, ["other"]])
        single = once.search(["colchicine"], top_n=1)[0].score
        repeated = many.search(["colchicine"], top_n=1)[0].score
        assert repeated > single
        assert repeated < 10 * single

    def test_a_common_term_scores_below_a_rare_one(self) -> None:
        """IDF: 'covid19' is in most of this corpus, 'colchicine' in one doc."""
        docs = [["covid19", "colchicine"]] + [["covid19", "other"]] * 9
        model = BM25Baseline(docs)
        rare = model.score(["colchicine"])[0]
        common = model.score(["covid19"])[0]
        assert rare > common

    def test_idf_is_never_negative(self) -> None:
        """A term in almost every document must not actively demote its holders.

        With the unsmoothed Robertson IDF this goes negative above 50% document
        frequency — on a COVID-only corpus that would happen to 'covid19'.
        """
        docs = [["covid19"]] * 10
        model = BM25Baseline(docs)
        assert float(model._idf.min()) >= 0.0

    def test_shorter_document_wins_on_equal_term_counts(self) -> None:
        """Length normalisation: the same hit in less text is more about it."""
        model = BM25Baseline([["lung"] + ["filler"] * 50, ["lung"]])
        hits = model.search(["lung"], top_n=2)
        assert hits[0].row_id == 1

    def test_b_zero_disables_length_normalisation(self) -> None:
        model = BM25Baseline([["lung"] + ["filler"] * 50, ["lung"]], b=0.0)
        scores = model.score(["lung"])
        assert scores[0] == pytest.approx(scores[1])


class TestSearch:
    def test_exact_term_ranks_its_document_first(self, bm25: BM25Baseline) -> None:
        assert bm25.search(["colchicine"], top_n=5)[0].row_id == 2

    def test_hits_are_ordered_by_descending_score(self, bm25: BM25Baseline) -> None:
        hits = bm25.search(["covid19", "vaccine"], top_n=5)
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    def test_top_n_caps_the_result_count(self, bm25: BM25Baseline) -> None:
        assert len(bm25.search(["covid19"], top_n=1)) == 1

    def test_row_ids_index_the_corpus(self, bm25: BM25Baseline) -> None:
        for hit in bm25.search(["vaccine"], top_n=5):
            assert 0 <= hit.row_id < len(DOCS)

    def test_only_matching_documents_are_returned(self, bm25: BM25Baseline) -> None:
        """No padding with zero-scoring documents."""
        assert all(h.score > 0 for h in bm25.search(["colchicine"], top_n=5))


class TestDegradation:
    def test_unknown_term_returns_no_hits(self, bm25: BM25Baseline) -> None:
        assert bm25.search(["notinvocabulary"], top_n=5) == []

    def test_empty_query_returns_no_hits(self, bm25: BM25Baseline) -> None:
        assert bm25.search([], top_n=5) == []

    def test_empty_corpus_builds_without_raising(self) -> None:
        assert BM25Baseline([]).search(["covid19"], top_n=5) == []

    def test_documents_with_no_tokens_are_tolerated(self) -> None:
        model = BM25Baseline([["lung"], [], ["vaccine"]])
        assert [h.row_id for h in model.search(["lung"], top_n=3)] == [0]

    def test_score_returns_one_value_per_document(self, bm25: BM25Baseline) -> None:
        assert bm25.score(["covid19"]).shape == (len(DOCS),)
