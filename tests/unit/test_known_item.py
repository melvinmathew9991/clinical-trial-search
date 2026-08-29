"""Known-item retrieval by registry id.

The capability this pins did not exist. Measured over 60 real ids drawn evenly
from all twelve registries, the shipped union retriever returned the requested
trial 0 times -- not at rank 1, not in the top 10, not at all. Trial ids live in
a column retrieval never read, and only 71 of 10,666 abstracts contain any
registry code, so no amount of ranking work could have reached it.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from medsearch.search.engine import SearchResponse, SearchResult
from medsearch.search.known_item import (
    KnownItemRetriever,
    TrialIdIndex,
    normalize_trial_id,
)


@pytest.fixture
def corpus_frame() -> pd.DataFrame:
    """Ids in the shapes the real registries actually use."""
    return pd.DataFrame(
        {
            "trial_id": [
                "NCT04372368",
                "CTRI/2021/05/033883",
                "2021-001036-25",
                "ChiCTR2000029739",
            ],
            "title": ["Convalescent plasma", "Yoga", "Vaccine", "Hydrogen nebuliser"],
            "abstract": ["a", "b", "c", "d"],
            "publication_date": ["2021-01-01"] * 4,
        }
    )


class _StubInner:
    """Returns a fixed ranking that never contains the queried trial."""

    def __init__(self, trial_ids: list[str]) -> None:
        self._trial_ids = trial_ids
        self.calls: list[dict[str, Any]] = []
        self.size = 4

    def search(self, query: str, **kwargs: Any) -> SearchResponse:
        self.calls.append(kwargs)
        return SearchResponse(
            query=query,
            results=[
                SearchResult(
                    rank=i,
                    score=1.0 / i,
                    trial_id=tid,
                    title="t",
                    abstract="a",
                    publication_date="2021-01-01",
                )
                for i, tid in enumerate(self._trial_ids, start=1)
            ],
        )


class TestNormalisation:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("NCT04372368", "NCT04372368"),
            ("nct04372368", "NCT04372368"),
            ("CTRI/2021/05/033883", "CTRI202105033883"),
            ("2021-001036-25", "202100103625"),
            ("  ChiCTR2000029739  ", "CHICTR2000029739"),
        ],
    )
    def test_separators_and_case_carry_no_meaning(self, raw: str, expected: str) -> None:
        assert normalize_trial_id(raw) == expected


class TestTrialIdIndex:
    def test_finds_an_id_typed_exactly(self, corpus_frame: pd.DataFrame) -> None:
        assert TrialIdIndex(corpus_frame).lookup("NCT04372368") == "NCT04372368"

    def test_finds_an_id_typed_in_another_case(self, corpus_frame: pd.DataFrame) -> None:
        assert TrialIdIndex(corpus_frame).lookup("nct04372368") == "NCT04372368"

    def test_finds_a_slashed_id_however_it_is_punctuated(self, corpus_frame: pd.DataFrame) -> None:
        index = TrialIdIndex(corpus_frame)
        assert index.lookup("CTRI/2021/05/033883") == "CTRI/2021/05/033883"
        assert index.lookup("CTRI 2021 05 033883") == "CTRI/2021/05/033883"

    def test_finds_an_id_pasted_beside_other_words(self, corpus_frame: pd.DataFrame) -> None:
        assert TrialIdIndex(corpus_frame).lookup("results for NCT04372368") == "NCT04372368"

    def test_ordinary_prose_is_not_an_id(self, corpus_frame: pd.DataFrame) -> None:
        assert TrialIdIndex(corpus_frame).lookup("convalescent plasma trials") is None

    def test_a_short_token_is_never_treated_as_an_id(self) -> None:
        """Guards against a stray word colliding with a very short id."""
        frame = pd.DataFrame(
            {
                "trial_id": ["NL945"],
                "title": ["x"],
                "abstract": ["x"],
                "publication_date": ["2021-01-01"],
            }
        )
        # Exact whole-query match still works; an in-sentence match does not.
        assert TrialIdIndex(frame).lookup("NL945") == "NL945"
        assert TrialIdIndex(frame).lookup("trial NL945 outcomes") is None

    def test_colliding_ids_are_served_to_nobody(self) -> None:
        """Two ids reducing to one key cannot be resolved, so neither is."""
        frame = pd.DataFrame(
            {
                "trial_id": ["ABC-123456", "ABC123456"],
                "title": ["x", "y"],
                "abstract": ["x", "y"],
                "publication_date": ["2021-01-01"] * 2,
            }
        )
        assert TrialIdIndex(frame).lookup("ABC123456") is None


class TestKnownItemRetriever:
    def test_the_trial_itself_is_promoted_to_rank_one(self, corpus_frame: pd.DataFrame) -> None:
        inner = _StubInner(["ChiCTR2000029739", "2021-001036-25"])
        retriever = KnownItemRetriever(inner, corpus_frame)
        results = retriever.search("NCT04372368").results
        assert results[0].trial_id == "NCT04372368"
        assert results[0].rank == 1

    def test_the_ranking_below_it_is_preserved(self, corpus_frame: pd.DataFrame) -> None:
        """Citing trials keep their order, so citation search still works."""
        inner = _StubInner(["ChiCTR2000029739", "2021-001036-25"])
        retriever = KnownItemRetriever(inner, corpus_frame)
        results = retriever.search("NCT04372368").results
        assert [r.trial_id for r in results[1:]] == [
            "ChiCTR2000029739",
            "2021-001036-25",
        ]

    def test_ranks_stay_sequential(self, corpus_frame: pd.DataFrame) -> None:
        inner = _StubInner(["ChiCTR2000029739", "2021-001036-25"])
        retriever = KnownItemRetriever(inner, corpus_frame)
        results = retriever.search("NCT04372368").results
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_an_already_present_trial_is_moved_not_duplicated(
        self, corpus_frame: pd.DataFrame
    ) -> None:
        inner = _StubInner(["ChiCTR2000029739", "NCT04372368"])
        retriever = KnownItemRetriever(inner, corpus_frame)
        ids = [r.trial_id for r in retriever.search("NCT04372368").results]
        assert ids == ["NCT04372368", "ChiCTR2000029739"]

    def test_a_prose_query_is_left_completely_alone(self, corpus_frame: pd.DataFrame) -> None:
        inner = _StubInner(["ChiCTR2000029739", "2021-001036-25"])
        retriever = KnownItemRetriever(inner, corpus_frame)
        results = retriever.search("convalescent plasma").results
        assert [r.trial_id for r in results] == ["ChiCTR2000029739", "2021-001036-25"]

    def test_keyword_arguments_reach_the_wrapped_retriever(
        self, corpus_frame: pd.DataFrame
    ) -> None:
        """The wrapper stands in for either retriever, which take different keywords."""
        inner = _StubInner(["ChiCTR2000029739"])
        retriever = KnownItemRetriever(inner, corpus_frame)
        retriever.search("anything", per_method=7)
        assert inner.calls == [{"per_method": 7}]

    def test_other_attributes_delegate_to_the_wrapped_retriever(
        self, corpus_frame: pd.DataFrame
    ) -> None:
        inner = _StubInner([])
        assert KnownItemRetriever(inner, corpus_frame).size == 4
