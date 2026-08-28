"""Text normalisation primitives.

Every regex is compiled once at module import. The legacy implementation
called ``re.sub`` with a string pattern inside a per-document loop, paying
pattern parsing 10,666 times per field per run (PRD F-11).

The transforms themselves reproduce the legacy cleaning order exactly, so
embeddings stay comparable with the original project's results.
"""

from __future__ import annotations

import re
import string
from typing import Final

#: Applied first: strips handles, URLs, and any non-alphanumeric run.
#: Mirrors the legacy `remove_urls` pattern, which did all three at once.
_URL_AND_SYMBOL_RE: Final[re.Pattern[str]] = re.compile(
    r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"
)
_DIGIT_RE: Final[re.Pattern[str]] = re.compile(r"\d+")
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")

#: ``str.translate`` table for punctuation removal -- built once.
_PUNCTUATION_TABLE: Final[dict[int, int | None]] = str.maketrans("", "", string.punctuation)


def to_lowercase(text: str) -> str:
    """Lowercase the input."""
    return text.lower()


def strip_urls_and_symbols(text: str) -> str:
    """Remove URLs, @handles, and non-alphanumeric characters.

    Collapses the resulting whitespace so tokenisation does not produce
    empty tokens.
    """
    return _WHITESPACE_RE.sub(" ", _URL_AND_SYMBOL_RE.sub(" ", text)).strip()


def strip_digits(text: str) -> str:
    """Remove digit runs.

    Clinical abstracts are dense with dosages and sample sizes; the numbers
    carry no distributional signal for retrieval and inflate the vocabulary.
    """
    return _DIGIT_RE.sub("", text)


def strip_punctuation(text: str) -> str:
    """Remove ASCII punctuation."""
    return text.translate(_PUNCTUATION_TABLE)


def collapse_whitespace(text: str) -> str:
    """Normalise all whitespace -- newlines included -- to single spaces."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text: str) -> str:
    """Apply the full normalisation chain.

    Order matters and matches the legacy pipeline: lowercase, strip URLs and
    symbols, strip digits, strip punctuation, collapse whitespace.

    Args:
        text: Raw document or query text.

    Returns:
        Cleaned text, ready for tokenisation. Never ``None``.

    Example:
        >>> clean_text("COVID-19 affects 1,200 patients (see http://x.co)")
        'covid affects patients see'
    """
    if not text:
        return ""
    cleaned = to_lowercase(text)
    cleaned = strip_urls_and_symbols(cleaned)
    cleaned = strip_digits(cleaned)
    cleaned = strip_punctuation(cleaned)
    return collapse_whitespace(cleaned)
