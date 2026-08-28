"""Text normalisation primitives.

Every regex is compiled once at module import. The legacy implementation
called ``re.sub`` with a string pattern inside a per-document loop, paying
pattern parsing 10,666 times per field per run (PRD F-11).

Alphanumeric identity is preserved
----------------------------------

The legacy chain -- reproduced here until the pre-deployment domain audit --
stripped every digit, on the stated rationale that "the numbers carry no
distributional signal for retrieval". That is true of general prose and false
of biomedical text, where the digit *is* the identity:

===================  ==================  ==========================
Term                 Stripped (old)      Consequence
===================  ==================  ==========================
``CD4`` / ``CD8``    ``cd`` / ``cd``     two T-cell markers, one token
``ACE2``             ``ace``             the SARS-CoV-2 entry receptor
``SARS-CoV-2``       ``sars``, ``cov``   strain number lost
``BNT162b2``         ``bntb``            the Pfizer vaccine
``NCT04508933``      ``nct``             *every* registry ID identical
``interleukin-6``    ``interleukin``     IL-6 indistinguishable from IL-1
===================  ==================  ==========================

A sample of 4,000 abstracts contained **6,077** such tokens. For a clinical
trial search engine this removed precisely the tokens that identify an
intervention.

The chain now keeps digits bound to letters, and joins intra-word hyphens so
``sars-cov-2`` becomes one token rather than three fragments. Bare numerals
survive normalisation but are dropped downstream by ``min_token_length``,
which removes single digits while keeping ``19`` in ``covid19``.
"""

from __future__ import annotations

import re
from typing import Final

#: URLs and @handles, removed before anything else so their punctuation does
#: not survive as fragments.
_URL_RE: Final[re.Pattern[str]] = re.compile(r"\w+:\/\/\S+|@[A-Za-z0-9_]+")

#: An intra-word hyphen joins rather than splits: ``sars-cov-2`` -> ``sarscov2``,
#: ``interleukin-6`` -> ``interleukin6``. Splitting these produced fragments
#: that collide across distinct entities; joining keeps one discriminative
#: token. Only hyphens *between* alphanumerics are joined, so a dash used as
#: punctuation still separates words.
_INTRAWORD_HYPHEN_RE: Final[re.Pattern[str]] = re.compile(
    # Escapes rather than literals: U+2010..U+2014 are visually identical to
    # ASCII "-" in source. Publishers really do emit en-dashes in abstracts,
    # so they must be matched, but they should not be invisible in the code.
    r"(?<=[A-Za-z0-9])[-\u2010\u2011\u2012\u2013\u2014](?=[A-Za-z0-9])"
)

#: A decimal point between digits, likewise joined: ``0.5`` -> ``05`` rather
#: than two tokens. Dose magnitudes are rarely the query term, but splitting
#: them adds noise for nothing.
_INTRAWORD_DOT_RE: Final[re.Pattern[str]] = re.compile(r"(?<=\d)[.,](?=\d)")

#: Everything that is not alphanumeric or whitespace becomes a space. Applied
#: after the joins above, so the characters they consume are already gone.
_NON_ALNUM_RE: Final[re.Pattern[str]] = re.compile(r"[^0-9A-Za-z \t]")

_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")


def to_lowercase(text: str) -> str:
    """Lowercase the input."""
    return text.lower()


def strip_urls(text: str) -> str:
    """Remove URLs and @handles."""
    return _URL_RE.sub(" ", text)


def join_intraword_marks(text: str) -> str:
    """Join hyphens and decimal points that sit inside a token.

    ``sars-cov-2`` -> ``sarscov2``; ``0.5`` -> ``05``. A hyphen with a space on
    either side is punctuation and is left for :func:`strip_symbols` to turn
    into a separator.
    """
    return _INTRAWORD_DOT_RE.sub("", _INTRAWORD_HYPHEN_RE.sub("", text))


def strip_symbols(text: str) -> str:
    """Replace every non-alphanumeric character with a space."""
    return _NON_ALNUM_RE.sub(" ", text)


def collapse_whitespace(text: str) -> str:
    """Normalise all whitespace -- newlines included -- to single spaces."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text: str) -> str:
    """Apply the full normalisation chain.

    Order matters: lowercase, strip URLs, join intra-word marks, replace
    remaining symbols with spaces, collapse whitespace. The joins must precede
    symbol removal or the characters they depend on are already gone.

    Args:
        text: Raw document or query text.

    Returns:
        Cleaned text, ready for tokenisation. Never ``None``.

    Example:
        >>> clean_text("COVID-19 and SARS-CoV-2 affect ACE2 (see http://x.co)")
        'covid19 and sarscov2 affect ace2 see'
        >>> clean_text("CD4 and CD8 counts")
        'cd4 and cd8 counts'
    """
    if not text:
        return ""
    cleaned = to_lowercase(text)
    cleaned = strip_urls(cleaned)
    cleaned = join_intraword_marks(cleaned)
    cleaned = strip_symbols(cleaned)
    return collapse_whitespace(cleaned)
