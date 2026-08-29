"""Free-standing negation as a query operator.

EVALUATION_AUDIT section 8 proved no additive feature scheme can express
negation: a representation can only add evidence, and negation needs evidence
subtracted. Bigrams were built and rejected on measurement. This is the
operator that replaced them, and these tests pin the three decisions that made
it work rather than the two that did not.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd
import pytest

from medsearch.search.engine import SearchResponse, SearchResult
from medsearch.search.negation import NegationFilter, asserts, parse_negation


class _StubInner:
    def __init__(self, trial_ids: list[str]) -> None:
        self._trial_ids = trial_ids
        self.calls: list[dict[str, Any]] = []

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


@pytest.fixture
def corpus_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trial_id": ["ASSERTS", "DENIES", "AVOIDS", "SILENT"],
            "title": ["a", "b", "c", "d"],
            "abstract": [
                "patients on invasive mechanical ventilation in the ICU",
                "patients managed without mechanical ventilation",
                "therapy that reduces the need for mechanical ventilation",
                "an unrelated vaccine trial",
            ],
            "publication_date": ["2021-01-01"] * 4,
        }
    )


class TestParsing:
    def test_a_cue_opens_a_scope(self) -> None:
        _, spans = parse_negation("patients not requiring supplemental oxygen")
        assert spans == [["requiring", "supplemental", "oxygen"]]

    def test_without_is_a_cue_too(self) -> None:
        _, spans = parse_negation("treatment without mechanical ventilation")
        assert spans == [["mechanical", "ventilation"]]

    def test_a_conjunction_closes_the_scope(self) -> None:
        """Negation does not carry across a clause boundary.

        "without mechanical ventilation but requiring oxygen" negates only the
        first clause; the oxygen requirement is asserted, not excluded.
        """
        _, spans = parse_negation("without mechanical ventilation but requiring oxygen")
        assert spans == [["mechanical", "ventilation"]]

    def test_the_query_itself_is_returned_unchanged(self) -> None:
        """The negated words still describe the topic, so retrieval keeps them.

        Stripping them leaves "patients", which retrieves nothing useful.
        """
        query = "patients not requiring supplemental oxygen"
        assert parse_negation(query)[0] == query

    def test_a_single_common_word_is_not_worth_negating(self) -> None:
        """The regression that took main-set Recall@10 from 0.702 to 0.698.

        "spread by people without symptoms" parsed to "exclude anything
        mentioning symptoms", which removed the asymptomatic-transmission
        trials the query was asking for. Such a phrase names a concept, not an
        exclusion.
        """
        assert parse_negation("spread by people without symptoms")[1] == []

    def test_prose_without_a_cue_parses_to_nothing(self) -> None:
        assert parse_negation("severe covid-19 pneumonia")[1] == []


class TestAssertionDetection:
    SPAN: ClassVar[list[str]] = ["mechanical", "ventilation"]

    def test_a_plain_mention_is_an_assertion(self) -> None:
        assert asserts("patients on invasive mechanical ventilation", self.SPAN)

    def test_a_negated_mention_is_not(self) -> None:
        assert not asserts("managed without mechanical ventilation", self.SPAN)

    def test_avoidance_language_is_not_an_assertion(self) -> None:
        """The finding that made this work at all.

        Five of six gold documents for "treatment without mechanical
        ventilation" phrase the negated sense as avoidance -- "reduces the need
        for", "decrease the need of". A cue list holding only grammatical
        negation classified every one of them as an assertion and removed them.
        """
        assert not asserts("therapy that reduces the need for mechanical ventilation", self.SPAN)
        assert not asserts("may decrease the need of mechanical ventilation", self.SPAN)
        assert not asserts("patients at risk of need for mechanical ventilation", self.SPAN)

    def test_absence_of_the_concept_is_not_an_assertion(self) -> None:
        assert not asserts("an unrelated vaccine trial", self.SPAN)

    def test_every_token_of_the_span_must_appear(self) -> None:
        assert not asserts("mechanical circulatory support", self.SPAN)


class TestNegationFilter:
    def test_a_query_without_a_cue_passes_straight_through(
        self, corpus_frame: pd.DataFrame
    ) -> None:
        inner = _StubInner(["ASSERTS", "DENIES"])
        retriever = NegationFilter(inner, corpus_frame)
        ids = [r.trial_id for r in retriever.search("covid pneumonia", top_n=10).results]
        assert ids == ["ASSERTS", "DENIES"]

    def test_documents_asserting_the_concept_are_removed(self, corpus_frame: pd.DataFrame) -> None:
        inner = _StubInner(["ASSERTS", "DENIES", "AVOIDS", "SILENT"])
        retriever = NegationFilter(inner, corpus_frame)
        ids = [
            r.trial_id
            for r in retriever.search("treatment without mechanical ventilation", top_n=10).results
        ]
        assert "ASSERTS" not in ids

    def test_documents_denying_or_avoiding_it_are_kept(self, corpus_frame: pd.DataFrame) -> None:
        """The documents a negated query wants are mostly the ones that mention
        the concept in order to deny it. A plain exclusion filter loses them."""
        inner = _StubInner(["ASSERTS", "DENIES", "AVOIDS", "SILENT"])
        retriever = NegationFilter(inner, corpus_frame)
        ids = [
            r.trial_id
            for r in retriever.search("treatment without mechanical ventilation", top_n=10).results
        ]
        assert "DENIES" in ids
        assert "AVOIDS" in ids

    def test_it_retrieves_deeper_before_filtering(self, corpus_frame: pd.DataFrame) -> None:
        """Otherwise filtering returns a short list rather than a different one."""
        inner = _StubInner(["SILENT"])
        retriever = NegationFilter(inner, corpus_frame, depth_multiplier=3)
        retriever.search("treatment without mechanical ventilation", top_n=10)
        assert inner.calls == [{"top_n": 30}]

    def test_ranks_are_renumbered_after_filtering(self, corpus_frame: pd.DataFrame) -> None:
        inner = _StubInner(["ASSERTS", "DENIES", "AVOIDS"])
        retriever = NegationFilter(inner, corpus_frame)
        results = retriever.search("treatment without mechanical ventilation", top_n=10).results
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_filtering_everything_away_explains_itself(self, corpus_frame: pd.DataFrame) -> None:
        inner = _StubInner(["ASSERTS"])
        retriever = NegationFilter(inner, corpus_frame)
        response = retriever.search("treatment without mechanical ventilation", top_n=10)
        assert response.is_empty
        assert response.reason is not None
        assert "excludes" in response.reason

    def test_other_attributes_delegate(self, corpus_frame: pd.DataFrame) -> None:
        inner = _StubInner([])
        inner.size = 4  # type: ignore[attr-defined]
        assert NegationFilter(inner, corpus_frame).size == 4
