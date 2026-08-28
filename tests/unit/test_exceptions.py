"""Tests for the exception hierarchy.

Rules.md section 4 makes two promises about errors, and both are load-bearing
rather than cosmetic:

1. **Every raised error descends from ``MedSearchError``.** The CLI and the
   Streamlit app both catch exactly that type and render ``str(exc)`` without
   a traceback. An error outside the hierarchy reaches the user as a stack
   dump instead.
2. **Messages state what failed, what was expected, and what to do.** These
   strings *are* the user interface for every failure path, so they are
   asserted here like any other output.

Nothing tested this module until the pre-deployment audit, which is how two
call sites came to raise bare ``ValueError`` for conditions the hierarchy
already had classes for.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from medsearch.exceptions import (
    ArtefactMismatchError,
    ConfigurationError,
    CorpusNotFoundError,
    DataError,
    EmptyCorpusError,
    IndexBuildError,
    MedSearchError,
    ModelError,
    ModelNotTrainedError,
    ResourceError,
    SchemaValidationError,
    StaleIndexError,
)

ALL_ERRORS = (
    ConfigurationError,
    DataError,
    CorpusNotFoundError,
    SchemaValidationError,
    EmptyCorpusError,
    ModelError,
    ModelNotTrainedError,
    ArtefactMismatchError,
    StaleIndexError,
    IndexBuildError,
    ResourceError,
)


class TestHierarchy:
    """The contract the CLI and app rely on when they catch MedSearchError."""

    @pytest.mark.parametrize("error", ALL_ERRORS)
    def test_every_error_is_a_medsearch_error(self, error: type[Exception]) -> None:
        assert issubclass(error, MedSearchError)

    @pytest.mark.parametrize(
        ("error", "parent"),
        [
            (CorpusNotFoundError, DataError),
            (SchemaValidationError, DataError),
            (EmptyCorpusError, DataError),
            (ModelNotTrainedError, ModelError),
            (ArtefactMismatchError, ModelError),
            (StaleIndexError, ModelError),
        ],
    )
    def test_subclasses_keep_their_documented_parent(
        self, error: type[Exception], parent: type[Exception]
    ) -> None:
        """Callers catch `DataError` or `ModelError` to handle a whole family."""
        assert issubclass(error, parent)

    def test_catching_the_base_catches_a_leaf(self) -> None:
        with pytest.raises(MedSearchError):
            raise ArtefactMismatchError(expected="a", actual="b")


class TestMessages:
    """Rules section 4: what failed, what was expected, what to do."""

    def test_corpus_not_found_names_the_path_and_the_fix(self) -> None:
        message = str(CorpusNotFoundError(Path("data/raw/missing.csv")))
        assert "missing.csv" in message
        assert "Expected:" in message
        assert "Fix:" in message

    def test_corpus_not_found_does_not_promise_a_dead_recovery_path(self) -> None:
        """`make data` migrated from Part_1, which no longer holds a corpus.

        The message used to send the reader after a file that the audit
        deleted. Pointing someone at a nonexistent recovery is worse than
        saying nothing.
        """
        message = str(CorpusNotFoundError(Path("data/raw/missing.csv")))
        assert "run `make data`" not in message

    def test_corpus_not_found_names_where_to_get_it(self) -> None:
        """Saying "place the export here" assumes the reader has the export."""
        message = str(CorpusNotFoundError(Path("data/raw/missing.csv")))
        assert "dimensions.figshare.com" in message

    def test_artefact_mismatch_shows_both_sides(self) -> None:
        message = str(ArtefactMismatchError(expected="model-abc", actual="index-xyz"))
        assert "model-abc" in message
        assert "index-xyz" in message
        assert "Fix:" in message

    def test_artefact_mismatch_keeps_its_operands(self) -> None:
        """Handlers branch on the values, not on parsing the message."""
        error = ArtefactMismatchError(expected="a", actual="b")
        assert (error.expected, error.actual) == ("a", "b")

    def test_empty_corpus_carries_a_message(self) -> None:
        """Raised by SIF weighting when the token cache holds nothing."""
        message = str(EmptyCorpusError("no tokens\n  Expected: one term\n  Fix: re-run"))
        assert "Expected:" in message
        assert "Fix:" in message


class TestTypedRaisesAtCallSites:
    """Regression tests for the two bare ValueErrors the audit found."""

    def test_dimension_mismatch_raises_artefact_mismatch(self) -> None:
        """`engine.py` compared index and model dims and raised ValueError.

        Same wrong-pairing failure the fingerprint check guards, detected by
        shape instead — so it must be the same typed error, or the CLI shows a
        traceback for one and a clean message for the other.
        """
        import inspect

        from medsearch.search import engine

        source = inspect.getsource(engine)
        assert "raise ValueError" not in source
        assert "ArtefactMismatchError" in source

    def test_sif_weighting_raises_empty_corpus(self) -> None:
        import inspect

        from medsearch.embeddings import weighting

        source = inspect.getsource(weighting)
        assert "raise ValueError" not in source
        assert "EmptyCorpusError" in source
