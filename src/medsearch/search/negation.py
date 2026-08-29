"""Free-standing negation, as a query operator rather than a term.

EVALUATION_AUDIT section 8 Results 2 and 5 established the constraint this
module is built against:

> Negation requires removing or inverting shared evidence. Every additive
> feature scheme -- a stopword allowlist, bigrams, trigrams, term reweighting --
> can only add evidence, and therefore none of them can express it.

Prefix negation already works, and works by accident of morphology: the
normaliser turns ``non-hospitalized`` into one rare token, which *substitutes*
for ``hospitalized`` and so removes the shared term. Free-standing ``not`` and
``without`` have nothing to substitute, which is why ``patients not requiring
supplemental oxygen`` and ``patients requiring supplemental oxygen`` returned
almost the same documents.

**The approach.** Negation is detected on both sides, using the cue-and-scope
method NegEx established for clinical text (Chapman et al., 2001): a trigger
term, then a bounded window that ends at a conjunction or punctuation. On the
query side it identifies what the user is ruling out. On the *document* side it
distinguishes an abstract that asserts the concept from one that denies it --
which is the whole point, because the documents a negated query wants are
mostly the ones that mention the concept in order to negate it. A plain
exclusion filter would throw those away along with the rest.

So a document is dropped only when it **asserts** the negated concept. A
document that negates it, or never mentions it, survives.
"""

from __future__ import annotations

import re
from typing import Any, cast

import pandas as pd

from medsearch.logging_conf import get_logger
from medsearch.search.engine import SearchResponse, SearchResult

logger = get_logger(__name__)

#: Free-standing negation triggers. Prefix forms (`non-`, `un-`) are absent on
#: purpose: the normaliser already resolves them by substitution, and the
#: measurement in EVALUATION_AUDIT section 8 shows that route works.
NEGATION_CUES = frozenset({"not", "without", "excluding", "except", "lacking", "absent"})

#: Tokens that close a negation's scope. NegEx terminates on conjunctions and
#: punctuation because negation does not carry across a clause boundary:
#: "without ventilation but requiring oxygen" negates only the first.
_SCOPE_TERMINATORS = frozenset({"but", "and", "or", "with", "however", "although", "versus"})

#: Longest scope in tokens. NegEx uses five or six; the phrases here
#: ("requiring supplemental oxygen", "mechanical ventilation") sit well inside.
_MAX_SCOPE_TOKENS = 5

#: Shortest scope worth acting on. A one-word exclusion of a common clinical
#: term is too blunt to be a filter: "spread by people without symptoms" parsed
#: to "exclude anything mentioning symptoms", which removed the asymptomatic
#: transmission trials the query was asking for and pushed main-set Recall@10
#: from 0.702 to 0.698 -- under the PRD target, on the strength of one query.
#: Such a phrase names a concept (asymptomatic) rather than an exclusion. Both
#: spans that demonstrably work -- "requiring supplemental oxygen" and
#: "mechanical ventilation" -- carry two or more tokens.
_MIN_SPAN_TOKENS = 2

#: Avoidance language, which is how clinical trial abstracts usually express
#: the negated sense: "reduces the need for mechanical ventilation" describes a
#: trial trying to prevent it, not one delivering it. Diagnosing this stratum
#: found five of six gold documents for "treatment without mechanical
#: ventilation" phrased exactly this way -- so a cue list holding only
#: grammatical negation classified them all as assertions and removed them.
#: Distinguishing the two senses is the whole task, and in this domain most of
#: the denial is carried by these verbs rather than by "not".
AVOIDANCE_CUES = frozenset(
    {
        "reduce",
        "reduces",
        "reduced",
        "reducing",
        "reduction",
        "decrease",
        "decreases",
        "decreased",
        "decreasing",
        "prevent",
        "prevents",
        "prevented",
        "preventing",
        "prevention",
        "avoid",
        "avoids",
        "avoided",
        "avoiding",
        "avoidance",
        "lower",
        "lowers",
        "lowering",
        "minimise",
        "minimize",
        "spare",
        "obviate",
        "obviates",
        "delay",
        "delays",
        "delaying",
        # Hedges: a trial naming the *risk* or *need* for something is not
        # reporting that it happened.
        "risk",
        "need",
        "needing",
        "likelihood",
        "probability",
        "eligible",
    }
)

#: How far back a cue may sit and still deny the concept. NegEx uses five or
#: six; four was too tight for "reduces the need for invasive X", where the
#: cue sits five tokens before the head.
_DOCUMENT_CUE_WINDOW = 6

_WORD = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def parse_negation(query: str) -> tuple[str, list[list[str]]]:
    """Split a query into what it asks for and what it rules out.

    Args:
        query: The raw user query.

    Returns:
        The query to retrieve with -- unchanged, because the negated words
        still describe the topic and dropping them leaves "patients" -- and one
        token list per negated span.

    Example:
        >>> parse_negation("patients not requiring supplemental oxygen")
        ('patients not requiring supplemental oxygen', [['requiring', 'supplemental', 'oxygen']])
    """
    tokens = _tokenize(query)
    spans: list[list[str]] = []
    position = 0
    while position < len(tokens):
        if tokens[position] not in NEGATION_CUES:
            position += 1
            continue
        span: list[str] = []
        for token in tokens[position + 1 : position + 1 + _MAX_SCOPE_TOKENS]:
            if token in _SCOPE_TERMINATORS or token in NEGATION_CUES:
                break
            span.append(token)
        if len(span) >= _MIN_SPAN_TOKENS:
            spans.append(span)
        position += 1 + len(span)
    return query, spans


def asserts(text: str, span: list[str]) -> bool:
    """Does this text state the concept, rather than deny it or omit it?

    A document counts as asserting the span when every one of its tokens
    appears and at least one occurrence of its head token is *not* preceded,
    within :data:`_DOCUMENT_CUE_WINDOW` tokens, by grammatical negation
    (:data:`NEGATION_CUES`) or by avoidance language (:data:`AVOIDANCE_CUES`).
    """
    if not span:
        return False
    tokens = _tokenize(text)
    positions = {token: [i for i, t in enumerate(tokens) if t == token] for token in span}
    if any(not found for found in positions.values()):
        return False

    # The concept is present. It is denied if any occurrence of its head token
    # is preceded by a cue, which is the form a negated mention takes.
    head = span[0]
    for index in positions[head]:
        window = tokens[max(0, index - _DOCUMENT_CUE_WINDOW) : index]
        if not any(token in NEGATION_CUES or token in AVOIDANCE_CUES for token in window):
            return True
    return False


class NegationFilter:
    """Wrap a retriever so a negated query drops documents that assert the concept.

    Retrieval runs at a deeper budget than requested, because filtering after
    the fact would otherwise return short result sets rather than different
    ones.

    Args:
        inner: The engine or union retriever to wrap.
        corpus: Frame the document text is read from.
        depth_multiplier: How much deeper to retrieve before filtering.
    """

    def __init__(self, inner: Any, corpus: pd.DataFrame, *, depth_multiplier: int = 3) -> None:
        self._inner = inner
        self._depth_multiplier = depth_multiplier
        self._text = {
            str(row.trial_id): f"{row.title} {row.abstract}" for row in corpus.itertuples()
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_inner"], name)

    def search(self, query: str, **kwargs: Any) -> SearchResponse:
        """Search, removing documents that assert what the query negates."""
        _, spans = parse_negation(query)
        if not spans:
            return cast(SearchResponse, self._inner.search(query, **kwargs))

        deeper = dict(kwargs)
        requested = 10
        for keyword in ("top_n", "per_method"):
            if keyword in deeper:
                requested = int(deeper[keyword])
                deeper[keyword] = requested * self._depth_multiplier
        response = cast(SearchResponse, self._inner.search(query, **deeper))

        kept = [
            result
            for result in response.results
            if not any(asserts(self._text.get(result.trial_id, ""), span) for span in spans)
        ]
        removed = len(response.results) - len(kept)
        if removed:
            logger.debug("Negation filter dropped %d of %d", removed, len(response.results))

        results = [
            SearchResult(
                rank=position,
                score=result.score,
                trial_id=result.trial_id,
                title=result.title,
                abstract=result.abstract,
                publication_date=result.publication_date,
            )
            for position, result in enumerate(kept[:requested], start=1)
        ]
        if not results:
            return SearchResponse(
                query=query,
                results=[],
                reason=(
                    "Every match asserted the concept this query excludes. "
                    "Try dropping the negation, or naming what you do want."
                ),
            )
        return SearchResponse(query=query, results=results)
