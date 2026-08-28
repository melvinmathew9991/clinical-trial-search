"""Corpus loading, validation, and fingerprinting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from medsearch.data.loader import corpus_fingerprint, iter_text, load_corpus
from medsearch.exceptions import (
    CorpusNotFoundError,
    EmptyCorpusError,
    SchemaValidationError,
)

SAMPLE_ROWS = 20


class TestLoadCorpus:
    def test_loads_every_row_by_default(self, corpus_csv: Path) -> None:
        assert len(load_corpus(corpus_csv)) == SAMPLE_ROWS

    def test_columns_are_canonical(self, corpus_csv: Path) -> None:
        frame = load_corpus(corpus_csv)
        assert {"trial_id", "title", "abstract", "publication_date"} <= set(frame.columns)

    def test_unused_columns_are_not_read(self, corpus_csv: Path) -> None:
        frame = load_corpus(corpus_csv)
        assert "Brief title" not in frame.columns

    def test_index_is_reset_to_positional(self, corpus_csv: Path) -> None:
        # Index position is the document id used throughout the search index,
        # so it must be a clean 0..n-1 range.
        frame = load_corpus(corpus_csv)
        assert list(frame.index) == list(range(len(frame)))

    def test_limit_caps_rows(self, corpus_csv: Path) -> None:
        assert len(load_corpus(corpus_csv, limit=5)) == 5

    def test_limit_larger_than_corpus_is_safe(self, corpus_csv: Path) -> None:
        assert len(load_corpus(corpus_csv, limit=10_000)) == SAMPLE_ROWS

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusNotFoundError, match="not found"):
            load_corpus(tmp_path / "absent.csv")

    def test_missing_file_message_is_actionable(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusNotFoundError) as exc_info:
            load_corpus(tmp_path / "absent.csv")
        assert "Fix:" in str(exc_info.value)

    def test_missing_required_column_raises(self, corpus_missing_abstract: Path) -> None:
        with pytest.raises(SchemaValidationError) as exc_info:
            load_corpus(corpus_missing_abstract)
        assert "Abstract" in str(exc_info.value)

    def test_schema_error_lists_what_was_found(self, corpus_missing_abstract: Path) -> None:
        with pytest.raises(SchemaValidationError) as exc_info:
            load_corpus(corpus_missing_abstract)
        assert "Found:" in str(exc_info.value)

    def test_corpus_with_no_text_raises(self, empty_corpus: Path) -> None:
        with pytest.raises(EmptyCorpusError, match="no rows"):
            load_corpus(empty_corpus)

    def test_embedded_newlines_do_not_break_parsing(self, tmp_path: Path) -> None:
        # Real abstracts contain newlines inside quoted fields.
        path = tmp_path / "multiline.csv"
        path.write_text(
            "Date added,Trial ID,Title,Abstract,Publication date\n"
            '2021-06-01,NCT1,A title,"line one\nline two\nline three",2021-06-01\n',
            encoding="utf-8",
        )
        frame = load_corpus(path)
        assert len(frame) == 1
        assert "line two" in frame["abstract"].iloc[0]

    def test_optional_columns_may_be_absent(self, tmp_path: Path) -> None:
        path = tmp_path / "minimal.csv"
        path.write_text(
            "Trial ID,Title,Abstract,Publication date\nNCT1,T,An abstract,2021-06-01\n",
            encoding="utf-8",
        )
        assert len(load_corpus(path)) == 1


class TestLimitWarning:
    def test_sampling_is_announced(
        self, corpus_csv: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            load_corpus(corpus_csv, limit=3)
        assert "SAMPLED CORPUS" in caplog.text

    def test_full_load_does_not_warn(
        self, corpus_csv: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            load_corpus(corpus_csv)
        assert "SAMPLED CORPUS" not in caplog.text


class TestFingerprint:
    def test_is_stable_across_calls(self, corpus_csv: Path) -> None:
        assert corpus_fingerprint(corpus_csv) == corpus_fingerprint(corpus_csv)

    def test_is_short(self, corpus_csv: Path) -> None:
        assert len(corpus_fingerprint(corpus_csv)) == 16

    def test_is_hexadecimal(self, corpus_csv: Path) -> None:
        int(corpus_fingerprint(corpus_csv), 16)

    def test_changes_when_content_changes(self, corpus_csv: Path) -> None:
        before = corpus_fingerprint(corpus_csv)
        corpus_csv.write_text(corpus_csv.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        assert corpus_fingerprint(corpus_csv) != before

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CorpusNotFoundError):
            corpus_fingerprint(tmp_path / "absent.csv")


class TestIterText:
    def test_returns_one_string_per_row(self, corpus_csv: Path) -> None:
        frame = load_corpus(corpus_csv)
        assert len(iter_text(frame, "abstract")) == len(frame)

    def test_values_are_strings(self, corpus_csv: Path) -> None:
        frame = load_corpus(corpus_csv)
        assert all(isinstance(t, str) for t in iter_text(frame, "abstract"))

    def test_title_field_also_works(self, corpus_csv: Path) -> None:
        frame = load_corpus(corpus_csv)
        assert len(iter_text(frame, "title")) == len(frame)

    def test_nulls_become_empty_strings(self) -> None:
        frame = pd.DataFrame({"abstract": ["text", None]})
        assert iter_text(frame, "abstract") == ["text", ""]

    def test_unknown_field_raises(self, corpus_csv: Path) -> None:
        frame = load_corpus(corpus_csv)
        with pytest.raises(SchemaValidationError, match="nonexistent"):
            iter_text(frame, "nonexistent")

    def test_does_not_mutate_the_frame(self, corpus_csv: Path) -> None:
        frame = load_corpus(corpus_csv)
        before = frame["abstract"].tolist()
        iter_text(frame, "abstract")
        assert frame["abstract"].tolist() == before
