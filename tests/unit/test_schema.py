"""Corpus schema declarations."""

from __future__ import annotations

from medsearch.data.schema import COLUMN_MAP, REQUIRED_SOURCE_COLUMNS, CorpusSchema


class TestSourceColumns:
    def test_reads_far_fewer_columns_than_the_export_has(self) -> None:
        # The Dimensions export has 21 columns; reading all of them is what
        # cost the legacy loader ~700 MB resident (PRD F-03).
        assert len(CorpusSchema.source_columns()) <= 6

    def test_every_required_column_is_read(self) -> None:
        assert set(CorpusSchema.source_columns()) >= REQUIRED_SOURCE_COLUMNS

    def test_text_fields_are_reachable_after_renaming(self) -> None:
        canonical = set(CorpusSchema.canonical_columns())
        assert set(CorpusSchema.text_fields) <= canonical

    def test_result_columns_are_reachable_after_renaming(self) -> None:
        canonical = set(CorpusSchema.canonical_columns())
        assert set(CorpusSchema.result_columns) <= canonical


class TestColumnMap:
    def test_renames_to_snake_case(self) -> None:
        assert COLUMN_MAP["Trial ID"] == "trial_id"
        assert COLUMN_MAP["Publication date"] == "publication_date"

    def test_no_canonical_name_contains_a_space(self) -> None:
        # Spaces in column names forced literals like df['Publication date']
        # through every legacy module.
        assert all(" " not in name for name in COLUMN_MAP.values())

    def test_canonical_names_are_unique(self) -> None:
        values = list(COLUMN_MAP.values())
        assert len(values) == len(set(values))

    def test_canonical_names_are_lowercase(self) -> None:
        assert all(name == name.lower() for name in COLUMN_MAP.values())


class TestMissingRequired:
    def test_none_missing_when_all_present(self) -> None:
        assert CorpusSchema.missing_required(CorpusSchema.source_columns()) == set()

    def test_reports_the_absent_column(self) -> None:
        present = [c for c in CorpusSchema.source_columns() if c != "Abstract"]
        assert CorpusSchema.missing_required(present) == {"Abstract"}

    def test_reports_all_absent_columns(self) -> None:
        assert CorpusSchema.missing_required([]) == set(REQUIRED_SOURCE_COLUMNS)

    def test_extra_columns_are_not_a_problem(self) -> None:
        extra = [*CorpusSchema.source_columns(), "Funder Country", "Registry"]
        assert CorpusSchema.missing_required(extra) == set()

    def test_accepts_any_iterable(self) -> None:
        assert CorpusSchema.missing_required(iter(CorpusSchema.source_columns())) == set()


class TestDtypes:
    def test_text_columns_are_declared_as_string(self) -> None:
        assert CorpusSchema.dtypes["Abstract"] == "string"

    def test_dtype_keys_are_all_read(self) -> None:
        assert set(CorpusSchema.dtypes) <= set(CorpusSchema.source_columns())
