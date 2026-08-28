"""Union retrieval: widen the result set instead of reranking it.

The evaluation (PRD §8.3) established two facts that look contradictory:

* **Rank fusion is worthless here.** Reciprocal Rank Fusion over the TF-IDF and
  embedding rankings scores 0.648 against TF-IDF's 0.648 — a difference of
  +0.0005, p = 0.98.
* **The methods are genuinely complementary.** 326 of the 496 relevant
  documents the embeddings retrieve (66%) are ones TF-IDF never returns, and
  the union of both top-10 lists reaches 0.955 recall against depth-matched
  TF-IDF's 0.715.

They reconcile once you notice the budget: at ten results a fusion must *drop*
a TF-IDF hit to admit an embedding hit, and TF-IDF's are more often relevant.
The complementarity is real and unusable by reranking at the same time.

So this module does not rerank. It returns the **union** of both top-*n* lists —
about twice as many documents — trading precision for a large, measured recall
gain. That is a product decision about how many trials to put in front of a
researcher, not a modelling one.

Ordering within the union is by Reciprocal Rank Fusion. It does not affect
recall at the full budget, but it puts documents both methods agree on first,
which is what a reader wants to see at the top.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from medsearch.logging_conf import get_logger
from medsearch.search.baseline import TfidfBaseline
from medsearch.search.engine import SearchEngine, SearchResponse, SearchResult

logger = get_logger(__name__)

#: RRF damping constant. 60 is the value from Cormack et al. (2009) and the
#: usual default; results are insensitive to it over a wide range.
RRF_K = 60


@dataclass(frozen=True, slots=True)
class UnionHit:
    """One document in the union, with its provenance."""

    trial_id: str
    score: float
    found_by: tuple[str, ...]

    @property
    def is_consensus(self) -> bool:
        """True when both retrievers returned this document."""
        return len(self.found_by) > 1


class UnionRetriever:
    """Return the union of an embedding ranking and a TF-IDF ranking.

    Args:
        engine: A loaded :class:`~medsearch.search.engine.SearchEngine`.
        baseline: A :class:`~medsearch.search.baseline.TfidfBaseline` built over
            the same corpus and the same preprocessing.
        corpus: Frame from :func:`~medsearch.data.loader.load_corpus`, used to
            resolve row ids into displayable records.

    Example:
        >>> retriever = UnionRetriever(engine, baseline, corpus)
        >>> response = retriever.search("lung failure", per_method=10)
        >>> len(response.results) <= 20
        True
    """

    def __init__(self, engine: SearchEngine, baseline: TfidfBaseline, corpus: pd.DataFrame) -> None:
        self._engine = engine
        self._baseline = baseline
        self._corpus = corpus
        self._trial_ids: list[str] = corpus["trial_id"].astype(str).tolist()

    @property
    def size(self) -> int:
        """Number of searchable documents."""
        return self._engine.size

    def _rank(self, query: str, per_method: int) -> list[UnionHit]:
        """Merge both rankings into one RRF-ordered union."""
        scores: dict[str, float] = {}
        sources: dict[str, set[str]] = {}

        response = self._engine.search(query, top_n=per_method)
        for rank, result in enumerate(response.results, start=1):
            scores[result.trial_id] = scores.get(result.trial_id, 0.0) + 1.0 / (RRF_K + rank)
            sources.setdefault(result.trial_id, set()).add("embedding")

        tokens = self._engine.preprocess(query)
        for rank, hit in enumerate(self._baseline.search(tokens, top_n=per_method), start=1):
            trial_id = self._trial_ids[hit.row_id]
            scores[trial_id] = scores.get(trial_id, 0.0) + 1.0 / (RRF_K + rank)
            sources.setdefault(trial_id, set()).add("keyword")

        return [
            UnionHit(trial_id=tid, score=score, found_by=tuple(sorted(sources[tid])))
            for tid, score in sorted(scores.items(), key=lambda kv: -kv[1])
        ]

    def search(self, query: str, *, per_method: int = 10) -> SearchResponse:
        """Search both retrievers and return their union.

        Args:
            query: Free-text query.
            per_method: Documents taken from each retriever. The union holds at
                most ``2 * per_method`` and usually fewer, since the two
                rankings overlap.

        Returns:
            A :class:`~medsearch.search.engine.SearchResponse` whose results are
            RRF-ordered. Empty with a reason when neither retriever matches --
            the same contract as the embedding engine alone.
        """
        hits = self._rank(query, per_method)
        if not hits:
            return SearchResponse(
                query=query,
                results=[],
                reason=(
                    "Neither semantic nor keyword search matched this query. It may be "
                    "all stopwords, or use terms absent from the corpus."
                ),
            )

        by_id = {str(row.trial_id): row for row in self._corpus.itertuples()}
        results: list[SearchResult] = []
        for rank, hit in enumerate(hits, start=1):
            row = by_id.get(hit.trial_id)
            if row is None:  # pragma: no cover - ids come from the same frame
                continue
            results.append(
                SearchResult(
                    rank=rank,
                    score=hit.score,
                    trial_id=hit.trial_id,
                    title=str(row.title),
                    abstract=str(row.abstract),
                    publication_date=str(row.publication_date),
                )
            )

        logger.debug(
            "Union search %r: %d results, %d found by both",
            query,
            len(results),
            sum(1 for h in hits if h.is_consensus),
        )
        return SearchResponse(query=query, results=results)

    def provenance(self, query: str, *, per_method: int = 10) -> list[UnionHit]:
        """Union hits with their source retrievers, for analysis and the UI."""
        return self._rank(query, per_method)
