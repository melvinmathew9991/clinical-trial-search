"""Corpus schema.

The Dimensions COVID-19 export has 21 columns. Six are used: two carry text
signal, four are display metadata. The remaining 15 are never read -- that is
the point of :meth:`CorpusSchema.source_columns`, which feeds ``usecols`` and
keeps a 29 MB CSV under ~90 MB resident instead of the ~700 MB the legacy
full-frame read cost (PRD F-03).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

#: Source column name -> canonical snake_case name.
#: The legacy code used the raw names with spaces throughout, which forced
#: string literals like ``df['Publication date']`` into every module.
COLUMN_MAP: Final[dict[str, str]] = {
    "Trial ID": "trial_id",
    "Title": "title",
    "Abstract": "abstract",
    "Publication date": "publication_date",
    "Date added": "date_added",
    "Phase": "phase",
}

#: Columns without which the pipeline cannot run.
REQUIRED_SOURCE_COLUMNS: Final[frozenset[str]] = frozenset(
    {"Trial ID", "Title", "Abstract", "Publication date"}
)

#: Text columns that can be embedded and searched.
TEXT_FIELDS: Final[tuple[str, ...]] = ("abstract", "title")

#: Columns returned in a search result, in display order.
RESULT_COLUMNS: Final[tuple[str, ...]] = (
    "trial_id",
    "title",
    "abstract",
    "publication_date",
)

#: Read text columns as ``str`` rather than letting pandas infer ``object``
#: with mixed types; avoids a second pass and surprise NaN floats in text.
SOURCE_DTYPES: Final[dict[str, str]] = {
    "Trial ID": "string",
    "Title": "string",
    "Abstract": "string",
    "Publication date": "string",
}


class CorpusSchema:
    """Declarative description of the clinical-trial corpus."""

    column_map = COLUMN_MAP
    required = REQUIRED_SOURCE_COLUMNS
    text_fields = TEXT_FIELDS
    result_columns = RESULT_COLUMNS
    dtypes = SOURCE_DTYPES

    @classmethod
    def source_columns(cls) -> list[str]:
        """Source column names to pass to ``pandas.read_csv(usecols=...)``."""
        return list(cls.column_map)

    @classmethod
    def canonical_columns(cls) -> list[str]:
        """Column names after renaming."""
        return list(cls.column_map.values())

    @classmethod
    def missing_required(cls, present: Iterable[str]) -> set[str]:
        """Return required source columns absent from ``present``."""
        return set(cls.required) - set(present)
