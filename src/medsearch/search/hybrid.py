"""Union retrieval: widen the result set instead of reranking it.

The evaluation established two facts that look contradictory:

* **Rank fusion cannot beat the lexical ranker at a fixed budget.** Truncated
  to ten documents the union scores nDCG@10 0.789 against BM25's 0.799 and
  TF-IDF's 0.797 — level, not ahead.
* **The methods are genuinely complementary.** The union of both top-10 lists
  reaches Recall@10 **0.702** and R-precision **0.639** on the re-judged eval
  set, against BM25's 0.471 and 0.458.

They reconcile once you notice the budget: at ten results a fusion must *drop*
a lexical hit to admit an embedding hit, and the lexical hits are more often
relevant. The complementarity is real and unusable by reranking at the same
time.

So this module does not rerank. It returns the **union** of both top-*n* lists —
about 17.8 documents rather than 10 — buying that recall with a wider result
set. That is a product decision about how many trials to put in front of a
researcher, not a modelling one. See PRD §8.4.

Ordering within the union is by weighted Reciprocal Rank Fusion. It does not
affect recall at the full budget, but it decides what a reader sees first, and
two details of it are load-bearing — see :data:`KEYWORD_WEIGHT` and
:meth:`UnionRetriever._rank`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from medsearch.logging_conf import get_logger
from medsearch.search.baseline import TfidfBaseline
from medsearch.search.engine import SearchEngine, SearchResponse, SearchResult

logger = get_logger(__name__)

#: RRF damping constant. 60 is the value from Cormack et al. (2009) and the
#: usual default; results are insensitive to it over a wide range.
RRF_K = 60

#: Weight on the keyword run's RRF contribution. The lexical ranker wins
#: Precision@1 in every measured stratum -- entity 0.909 vs 0.636, code 1.000
#: vs 0.000, negation 0.625 vs 0.250 -- so an unweighted fusion gives the
#: embedding run more say at the top than it has earned. A sweep over
#: 1.0/1.5/2.0/3.0 saturates at 1.5 (2.0 scores identically, 3.0 slightly
#: worse): main-set nDCG@10 0.753 -> 0.762 and R-precision 0.622 -> 0.639, with
#: Recall@10 unchanged at 0.702 because reordering cannot change the set.
KEYWORD_WEIGHT = 1.5


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

    #: Tie-break order: consensus first, then keyword-only, then
    #: embedding-only. Two runs that share no documents award identical RRF
    #: scores at identical ranks, and a plain ``sorted`` breaks that tie by
    #: insertion order -- which handed rank 1 to the embedding run on every
    #: known-item query in the round-3 ``code`` stratum, dropping its MRR@10 to
    #: 0.500 while the lexical ranker alone scored 1.000. This makes the
    #: fallback explicit and deterministic instead.
    _SOURCE_PRIORITY: ClassVar[dict[tuple[str, ...], int]] = {
        ("embedding", "keyword"): 0,
        ("keyword",): 1,
        ("embedding",): 2,
    }

    def _rank(self, query: str, per_method: int) -> list[UnionHit]:
        """Merge both rankings into one weighted-RRF union.

        Documents are ordered by descending fused score, ties broken by
        :attr:`_SOURCE_PRIORITY`. Consensus documents lead on score alone for
        any ``per_method`` up to 41; the tie-break is what makes the order below
        that threshold reproducible rather than dependent on dict insertion.
        """
        scores: dict[str, float] = {}
        sources: dict[str, set[str]] = {}

        response = self._engine.search(query, top_n=per_method)
        for rank, result in enumerate(response.results, start=1):
            scores[result.trial_id] = scores.get(result.trial_id, 0.0) + 1.0 / (RRF_K + rank)
            sources.setdefault(result.trial_id, set()).add("embedding")

        tokens = self._engine.preprocess(query)
        for rank, hit in enumerate(self._baseline.search(tokens, top_n=per_method), start=1):
            trial_id = self._trial_ids[hit.row_id]
            scores[trial_id] = scores.get(trial_id, 0.0) + KEYWORD_WEIGHT / (RRF_K + rank)
            sources.setdefault(trial_id, set()).add("keyword")

        def _order(item: tuple[str, float]) -> tuple[float, int]:
            trial_id, score = item
            return (-score, self._SOURCE_PRIORITY[tuple(sorted(sources[trial_id]))])

        return [
            UnionHit(trial_id=tid, score=score, found_by=tuple(sorted(sources[tid])))
            for tid, score in sorted(scores.items(), key=_order)
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
