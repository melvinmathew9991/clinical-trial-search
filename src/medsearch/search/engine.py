"""The search engine.

Ranking is one matrix-vector product against a pre-normalised index, then an
``argpartition`` for the top *n*.

The legacy implementation, per query, ran a Python loop of 10,666 calls to a
``cos_sim`` that recomputed ``norm(a)`` for the query every single time, then
sorted all 10,666 scores to take 10. This module does the same work in one
BLAS call plus an O(n) partial selection -- the difference between a laggy UI
and an instant one (ADR-003).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from medsearch._typing import FloatArray, IntArray
from medsearch.data.schema import CorpusSchema
from medsearch.embeddings.document import DocumentEmbedder
from medsearch.logging_conf import get_logger
from medsearch.preprocessing.pipeline import TextPreprocessor
from medsearch.search.index import DocumentIndex

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One ranked clinical trial."""

    rank: int
    score: float
    trial_id: str
    title: str
    abstract: str
    publication_date: str

    def truncated_abstract(self, limit: int = 300) -> str:
        """Abstract clipped for table display."""
        if len(self.abstract) <= limit:
            return self.abstract
        return self.abstract[:limit].rstrip() + "..."


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """The outcome of a search, successful or not.

    Carrying an explicit ``reason`` is how an all-out-of-vocabulary query
    stays a first-class, explainable outcome instead of the silent NaN the
    legacy code produced (Rules.md section 4).
    """

    query: str
    results: list[SearchResult]
    reason: str | None = None

    @property
    def is_empty(self) -> bool:
        """True when nothing was returned."""
        return not self.results

    def to_frame(self) -> pd.DataFrame:
        """Render as a DataFrame for display."""
        if not self.results:
            return pd.DataFrame(columns=["rank", "score", *CorpusSchema.result_columns])
        return pd.DataFrame(
            [
                {
                    "rank": r.rank,
                    "score": round(r.score, 4),
                    "trial_id": r.trial_id,
                    "title": r.title,
                    "abstract": r.abstract,
                    "publication_date": r.publication_date,
                }
                for r in self.results
            ]
        )


class SearchEngine:
    """Rank corpus documents by semantic similarity to a query.

    Args:
        index: A loaded, L2-normalised :class:`DocumentIndex`.
        embedder: Embedder built from the *same* model that produced ``index``.
        preprocessor: The same transform used on the corpus (PRD F-09).
        corpus: Frame from :func:`~medsearch.data.loader.load_corpus`, used to
            resolve row ids back into displayable records.

    Example:
        >>> engine = SearchEngine(index, embedder, preprocessor, corpus)
        >>> response = engine.search("lung failure", top_n=5)
        >>> [r.trial_id for r in response.results]
    """

    def __init__(
        self,
        index: DocumentIndex,
        embedder: DocumentEmbedder,
        preprocessor: TextPreprocessor,
        corpus: pd.DataFrame,
    ) -> None:
        if index.dim != embedder.dim:
            msg = (
                f"Index dimensionality ({index.dim}) does not match the model "
                f"({embedder.dim}). Rebuild the index."
            )
            raise ValueError(msg)

        self._index = index
        self._embedder = embedder
        self._preprocessor = preprocessor
        self._corpus = corpus

    @property
    def size(self) -> int:
        """Number of searchable documents."""
        return self._index.size

    def preprocess(self, query: str) -> list[str]:
        """Tokenize a query with the engine's preprocessor.

        Exposed so a companion retriever (the TF-IDF baseline in
        :mod:`medsearch.search.hybrid`) can guarantee identical preprocessing
        rather than constructing its own.
        """
        return self._preprocessor.transform(query)

    @property
    def is_sampled(self) -> bool:
        """True when the backing index covers only part of the corpus.

        Surfaced so the UI can say so rather than presenting partial coverage
        as complete.
        """
        return self._index.is_sampled

    def search(self, query: str, *, top_n: int = 10) -> SearchResponse:
        """Rank documents against ``query``.

        Args:
            query: Free-text search string.
            top_n: Maximum results to return.

        Returns:
            A :class:`SearchResponse`. Empty with a populated ``reason`` when
            the query is blank or contains no vectorisable token -- never a
            crash, never a NaN score (PRD F-26).
        """
        if not query or not query.strip():
            return SearchResponse(query=query, results=[], reason="Query is empty.")

        tokens = self._preprocessor.transform(query)
        if not tokens:
            return SearchResponse(
                query=query,
                results=[],
                reason=(
                    "Query reduced to nothing after preprocessing. It may be all "
                    "stopwords, digits, or punctuation."
                ),
            )

        query_vector = self._embedder.embed(tokens)

        # SIF requires the query to lose the same corpus direction the
        # documents lost at index-build time. Omitting this silently compares
        # vectors in different subspaces and quietly degrades every result.
        if self._index.principal_component is not None:
            from medsearch.embeddings.weighting import remove_component

            query_vector = remove_component(query_vector, self._index.principal_component)

        norm = float(np.linalg.norm(query_vector))
        if norm == 0.0:
            return SearchResponse(
                query=query,
                results=[],
                reason=(
                    f"None of {tokens} appear in the model vocabulary. "
                    f"Try a more common clinical term, or the FastText model, "
                    f"which can infer vectors for unseen words."
                ),
            )

        # Index rows are already unit-length, so normalising the query alone
        # makes the dot product exactly cosine similarity. One BLAS call.
        unit_query = (query_vector / norm).astype(np.float32, copy=False)
        scores = self._index.vectors @ unit_query

        top_indices = self._top_indices(scores, top_n)
        results = self._build_results(top_indices, scores)

        logger.debug("Query %r matched %d results", query, len(results))
        return SearchResponse(query=query, results=results)

    @staticmethod
    def _top_indices(scores: FloatArray, top_n: int) -> IntArray:
        """Indices of the ``top_n`` highest scores, best first.

        ``argpartition`` is O(n); a full ``argsort`` is O(n log n). Only the
        selected slice is then sorted (PRD F-29).
        """
        count = min(top_n, scores.shape[0])
        if count <= 0:
            return np.empty(0, dtype=np.int64)
        if count == scores.shape[0]:
            ordered: IntArray = np.argsort(scores)[::-1]
            return ordered
        partitioned = np.argpartition(scores, -count)[-count:]
        selected: IntArray = partitioned[np.argsort(scores[partitioned])[::-1]]
        return selected

    def _build_results(self, positions: IntArray, scores: FloatArray) -> list[SearchResult]:
        """Map index positions back to corpus records."""
        results: list[SearchResult] = []
        for rank, position in enumerate(positions, start=1):
            score = float(scores[position])
            # A zero score means an all-OOV document; it carries no signal and
            # is not worth showing.
            if score <= 0.0:
                continue
            row_id = int(self._index.row_ids[position])
            row = self._corpus.iloc[row_id]
            results.append(
                SearchResult(
                    rank=rank,
                    score=score,
                    trial_id=str(row.get("trial_id", "")),
                    title=str(row.get("title", "")),
                    abstract=str(row.get("abstract", "")),
                    publication_date=str(row.get("publication_date", "")),
                )
            )
        return results
