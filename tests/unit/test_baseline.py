"""Tests for the TF-IDF baseline.

This module started as a comparison yardstick and became load-bearing: it
outranks both embedding models on its own (Recall@10 0.648 vs 0.485) and it is
half of the union retrieval that ships (PRD section 8.4). Despite that it had
no test file of its own until the pre-deployment audit — only transitive
coverage through the evaluation and union tests.

The properties pinned here are the ones a silent change would break without
failing anything else: the sparse-only construction that keeps a 10,666 x
40,012 matrix inside 4.5 MB, the zero-norm guard that the legacy code lacked,
and the ranking contract the union depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

from medsearch.search.baseline import TfidfBaseline

DOCS: list[list[str]] = [
    ["covid", "vaccine", "trial"],
    ["covid", "vaccine", "efficacy", "trial"],
    ["colchicine", "inflammation"],
    ["remdesivir", "antiviral", "covid"],
]


@pytest.fixture
def baseline() -> TfidfBaseline:
    """A baseline over four short documents."""
    return TfidfBaseline(DOCS)


class TestConstruction:
    """Shape, dtype, and the sparsity that keeps the artefact small."""

    def test_matrix_covers_every_document_and_term(self, baseline: TfidfBaseline) -> None:
        vocabulary = {token for doc in DOCS for token in doc}
        assert baseline._matrix.shape == (len(DOCS), len(vocabulary))

    def test_matrix_stays_sparse(self, baseline: TfidfBaseline) -> None:
        """Densifying is the failure that matters at 40,012 terms.

        A dense float32 matrix of the real corpus would be 10,666 x 40,012 x 4
        bytes = 1.7 GB, past the whole serving budget. The measured artefact is
        4.5 MB.
        """
        import scipy.sparse as sp

        assert sp.issparse(baseline._matrix)

    def test_matrix_is_float32(self, baseline: TfidfBaseline) -> None:
        """float64 would double the footprint for no ranking benefit."""
        assert baseline._matrix.dtype == np.float32

    def test_rows_are_l2_normalised(self, baseline: TfidfBaseline) -> None:
        """Cosine similarity is then one dot product, as in ADR-003."""
        norms = np.sqrt(baseline._matrix.multiply(baseline._matrix).sum(axis=1)).ravel()
        assert np.allclose(np.asarray(norms).ravel(), 1.0, atol=1e-5)


class TestRanking:
    """The contract `UnionRetriever` consumes."""

    def test_exact_term_ranks_its_document_first(self, baseline: TfidfBaseline) -> None:
        hits = baseline.search(["colchicine"], top_n=4)
        assert hits[0].row_id == 2

    def test_hits_are_ordered_by_descending_score(self, baseline: TfidfBaseline) -> None:
        hits = baseline.search(["covid", "vaccine"], top_n=4)
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    def test_top_n_caps_the_result_count(self, baseline: TfidfBaseline) -> None:
        assert len(baseline.search(["covid"], top_n=2)) == 2

    def test_rarer_term_outranks_a_common_one(self, baseline: TfidfBaseline) -> None:
        """IDF is the whole reason this beats a dense average on rare terms."""
        hits = baseline.search(["efficacy", "covid"], top_n=4)
        assert hits[0].row_id == 1

    def test_row_ids_index_the_corpus(self, baseline: TfidfBaseline) -> None:
        """The union maps `row_id` through `trial_ids`; off-by-one is silent."""
        for hit in baseline.search(["covid"], top_n=4):
            assert 0 <= hit.row_id < len(DOCS)


class TestDegradation:
    """Where the legacy implementation produced NaN instead of an answer."""

    def test_unknown_term_returns_no_hits(self, baseline: TfidfBaseline) -> None:
        """Zero-norm query. The legacy path divided by it and ranked NaN."""
        assert baseline.search(["notinvocabulary"], top_n=4) == []

    def test_empty_query_returns_no_hits(self, baseline: TfidfBaseline) -> None:
        assert baseline.search([], top_n=4) == []

    def test_empty_corpus_builds_without_raising(self) -> None:
        """`doctor` inspects artefacts that may be mid-build; it must not crash."""
        empty = TfidfBaseline([])
        assert empty.search(["covid"], top_n=4) == []

    def test_documents_with_no_tokens_are_tolerated(self) -> None:
        built = TfidfBaseline([["covid"], [], ["vaccine"]])
        hits = built.search(["covid"], top_n=3)
        assert [h.row_id for h in hits] == [0]
