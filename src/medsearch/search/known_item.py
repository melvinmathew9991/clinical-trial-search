"""Known-item retrieval: answer a registry-ID query with the trial itself.

Every one of the 10,666 rows carries a unique ``Trial ID``, and retrieval runs
on ``abstract``. Nothing indexed the identifier, so the single most basic
operation a trial-search tool offers -- paste an ID, get that trial -- did not
work at all. Measured over 60 real IDs drawn evenly from all twelve registries,
the shipped union retriever returned the requested trial **0 times**: not at
rank 1, not inside the top 10, not anywhere in its result set.

This is not a ranking problem and is not fixed by ranking. An identifier is a
key, so it gets a lookup, and the lookup takes precedence over the ranking --
the same contract ClinicalTrials.gov, PubMed and the WHO ICTRP offer. Trials
whose abstracts *cite* the ID keep their places directly below, so the citation
behaviour that round 3's ``code`` stratum measured is preserved rather than
replaced.

Only 71 of 10,666 abstracts contain any registry code, so the text index could
never have supported this: the information lives in a column retrieval never
read.
"""

from __future__ import annotations

import re
from typing import Any, TypeAlias

import pandas as pd

from medsearch.logging_conf import get_logger
from medsearch.search.engine import SearchEngine, SearchResponse, SearchResult
from medsearch.search.hybrid import UnionRetriever
from medsearch.search.negation import NegationFilter

logger = get_logger(__name__)

#: The two things a user actually queries. A Protocol will not do: the two
#: take different keywords (``top_n`` against ``per_method``), which no single
#: structural signature satisfies.
_Retriever: TypeAlias = SearchEngine | UnionRetriever | NegationFilter

#: Registry identifiers are written inconsistently -- `CTRI/2021/05/033883`,
#: `ChiCTR2000029739`, `2021-001036-25`. Case and separators carry no meaning,
#: so both the stored ids and the query are reduced to letters and digits.
_NON_ALPHANUMERIC = re.compile(r"[^A-Za-z0-9]+")

#: A token must look like an identifier before it is treated as one: at least
#: this long, and containing a digit. Without both, ordinary words in a natural
#: language query would occasionally collide with an id.
_MIN_ID_LENGTH = 6


def normalize_trial_id(text: str) -> str:
    """Reduce an identifier to comparable form: upper case, letters and digits."""
    return _NON_ALPHANUMERIC.sub("", text).upper()


def _looks_like_an_id(token: str) -> bool:
    return len(token) >= _MIN_ID_LENGTH and any(character.isdigit() for character in token)


class TrialIdIndex:
    """Exact lookup from a registry identifier to its corpus row.

    Args:
        corpus: Frame from :func:`~medsearch.data.loader.load_corpus`.

    Example:
        >>> TrialIdIndex(corpus).lookup("nct04372368")
        'NCT04372368'
    """

    def __init__(self, corpus: pd.DataFrame) -> None:
        self._by_key: dict[str, str] = {}
        collisions = 0
        for trial_id in corpus["trial_id"].astype(str):
            key = normalize_trial_id(trial_id)
            if not key:
                continue
            if key in self._by_key and self._by_key[key] != trial_id:
                # Two distinct ids reducing to one key cannot be resolved, so
                # neither is served rather than serving the wrong one.
                collisions += 1
                self._by_key.pop(key, None)
                continue
            self._by_key[key] = trial_id
        if collisions:
            logger.warning(
                "%d trial ids collide after normalisation and are not looked up.", collisions
            )

    def __len__(self) -> int:
        return len(self._by_key)

    def lookup(self, query: str) -> str | None:
        """Return the trial id this query names, or ``None``.

        The whole query is tried first, so ``CTRI/2021/05/033883`` resolves
        despite its separators. Individual tokens are tried next, so an id
        pasted alongside other words still resolves.
        """
        whole = normalize_trial_id(query)
        if whole in self._by_key:
            return self._by_key[whole]
        for token in query.split():
            key = normalize_trial_id(token)
            if _looks_like_an_id(key) and key in self._by_key:
                return self._by_key[key]
        return None


class KnownItemRetriever:
    """Wrap a retriever so an identifier query answers with the trial itself.

    Delegates every other attribute to the wrapped retriever, so ``size``,
    ``is_sampled`` and the rest keep working and the wrapper can stand in
    wherever the inner object did.

    Args:
        inner: The engine or union retriever to wrap.
        corpus: Frame the identifiers and display fields are read from.
    """

    def __init__(self, inner: _Retriever, corpus: pd.DataFrame) -> None:
        self._inner = inner
        self._ids = TrialIdIndex(corpus)
        self._rows = {str(row.trial_id): row for row in corpus.itertuples()}

    def __getattr__(self, name: str) -> Any:
        # Only reached for attributes the wrapper does not define itself.
        return getattr(self.__dict__["_inner"], name)

    def search(self, query: str, **kwargs: Any) -> SearchResponse:
        """Search, promoting an identifier match to rank 1."""
        response = self._inner.search(query, **kwargs)
        trial_id = self._ids.lookup(query)
        if trial_id is None or trial_id not in self._rows:
            return response

        row = self._rows[trial_id]
        promoted = SearchResult(
            rank=1,
            # Ordinal only. A known-item hit is not scored by the ranker, and
            # this value is not comparable with the scores below it.
            score=1.0,
            trial_id=trial_id,
            title=str(row.title),
            abstract=str(row.abstract),
            publication_date=str(row.publication_date),
        )
        rest = [result for result in response.results if result.trial_id != trial_id]
        results = [promoted] + [
            SearchResult(
                rank=position,
                score=result.score,
                trial_id=result.trial_id,
                title=result.title,
                abstract=result.abstract,
                publication_date=result.publication_date,
            )
            for position, result in enumerate(rest, start=2)
        ]
        logger.debug("Known-item match for %r: %s promoted to rank 1", query, trial_id)
        return SearchResponse(query=query, results=results)
