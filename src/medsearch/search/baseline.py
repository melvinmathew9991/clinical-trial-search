"""TF-IDF keyword baseline.

The project's entire premise is that in-domain embeddings beat keyword search
on clinical text. That claim is worthless without a keyword search to compare
against, so this module provides one.

Implemented directly on ``scipy.sparse`` rather than pulling in scikit-learn:
the vocabulary is ~25,000 terms over ~10,666 documents, so a dense matrix
would be 25000 x 10666 x 4 bytes = **1.07 GB**. The sparse representation is
a few MB and fits the memory budget in Architecture.md section 9.

The baseline deliberately shares the *same* preprocessing as the embedding
path. Comparing an embedding model against a keyword model with different
tokenisation would measure the tokeniser, not the retrieval method.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from medsearch._typing import FloatArray
from medsearch.logging_conf import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BaselineHit:
    """One ranked document from the keyword baseline."""

    row_id: int
    score: float


class TfidfBaseline:
    """Classic TF-IDF with cosine similarity over an L2-normalised matrix.

    Uses the standard smoothed IDF, ``log((1 + n) / (1 + df)) + 1``, matching
    scikit-learn's default so the numbers are comparable with published
    baselines.

    Args:
        documents: Preprocessed token lists, in corpus row order.

    Example:
        >>> baseline = TfidfBaseline([["lung", "failure"], ["vaccine"]])
        >>> baseline.search(["lung"], top_n=1)[0].row_id
        0
    """

    def __init__(self, documents: Iterable[Sequence[str]]) -> None:
        docs = [list(d) for d in documents]
        self._n_documents = len(docs)

        vocabulary: dict[str, int] = {}
        for tokens in docs:
            for token in tokens:
                if token not in vocabulary:
                    vocabulary[token] = len(vocabulary)
        self._vocabulary = vocabulary

        if not vocabulary or not docs:
            self._matrix = sp.csr_matrix((self._n_documents, 0), dtype=np.float32)
            self._idf = np.zeros(0, dtype=np.float32)
            logger.warning("TF-IDF baseline built over an empty vocabulary")
            return

        # Build the term-frequency matrix in COO triplets -- never densely.
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        document_frequency = np.zeros(len(vocabulary), dtype=np.float64)

        for row, tokens in enumerate(docs):
            counts = Counter(tokens)
            for token, count in counts.items():
                index = vocabulary[token]
                rows.append(row)
                cols.append(index)
                values.append(float(count))
                document_frequency[index] += 1.0

        term_frequency = sp.csr_matrix(
            (values, (rows, cols)),
            shape=(self._n_documents, len(vocabulary)),
            dtype=np.float64,
        )

        # Smoothed IDF, as in scikit-learn's TfidfTransformer default.
        self._idf = (np.log((1.0 + self._n_documents) / (1.0 + document_frequency)) + 1.0).astype(
            np.float64
        )

        weighted = term_frequency.multiply(self._idf).tocsr()
        self._matrix = _l2_normalize_sparse(weighted).astype(np.float32)

        logger.info(
            "TF-IDF baseline: %d documents x %d terms, %.1f MB sparse",
            self._n_documents,
            len(vocabulary),
            self._matrix.data.nbytes / 1024**2,
        )

    @property
    def size(self) -> int:
        """Number of indexed documents."""
        return self._n_documents

    @property
    def vocabulary_size(self) -> int:
        """Number of distinct terms."""
        return len(self._vocabulary)

    def search(self, tokens: Sequence[str], *, top_n: int = 10) -> list[BaselineHit]:
        """Rank documents against a preprocessed query.

        Args:
            tokens: Query tokens, produced by the *same* preprocessor used on
                the corpus.
            top_n: Maximum hits to return.

        Returns:
            Hits ordered best first. Empty when no query term is in the
            vocabulary -- which is exactly the failure mode the embedding
            approach is meant to avoid, and therefore a result worth recording
            rather than an error.
        """
        known = [t for t in tokens if t in self._vocabulary]
        if not known or self._matrix.shape[1] == 0:
            return []

        query = np.zeros(self._matrix.shape[1], dtype=np.float64)
        for token, count in Counter(known).items():
            query[self._vocabulary[token]] = float(count)
        query *= self._idf

        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return []
        query /= norm

        scores: FloatArray = (self._matrix @ query.astype(np.float32)).astype(np.float32)

        count = min(top_n, scores.shape[0])
        positions = np.argpartition(scores, -count)[-count:]
        positions = positions[np.argsort(scores[positions])[::-1]]

        return [
            BaselineHit(row_id=int(p), score=float(scores[p])) for p in positions if scores[p] > 0
        ]


def _l2_normalize_sparse(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """L2-normalise each row of a sparse matrix, leaving empty rows at zero."""
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A.ravel()
    norms[norms == 0.0] = 1.0
    inverse = sp.diags(1.0 / norms)
    return (inverse @ matrix).tocsr()
