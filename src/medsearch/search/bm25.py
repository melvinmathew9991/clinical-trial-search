"""Okapi BM25 -- the keyword baseline the project should have started with.

Sprint 8 concluded that "a 40-line TF-IDF baseline beats both embedding
models". That conclusion held, but it was measured against the *weak* lexical
baseline. TF-IDF with cosine similarity is the 1970s formulation; BM25 has
been the standard since TREC-3 (1994) and is what any information-retrieval
reviewer will expect a new method to be compared against.

The distinction is not cosmetic on this corpus:

**Saturating term frequency.** TF-IDF scores a document containing
``colchicine`` twenty times as roughly twenty times more relevant than one
containing it once. BM25's ``tf / (tf + k1 · …)`` saturates, so the tenth
occurrence adds almost nothing. Trial abstracts repeat the intervention name
throughout; linear tf rewards verbosity rather than aboutness.

**Length normalisation with a tunable strength.** Cosine normalisation is
all-or-nothing. BM25's ``b`` interpolates between none and full, which matters
here because abstract lengths in this corpus vary by more than an order of
magnitude -- a 60-word registry stub against a 900-word structured abstract.

Parameters are the standard defaults, ``k1 = 1.2`` and ``b = 0.75``. They are
not tuned, deliberately: a baseline tuned on the same eval set it is scored on
stops being a baseline.

Implemented on ``scipy.sparse`` for the same reason as the TF-IDF path -- a
dense 31,189 x 10,666 float32 matrix would be 1.3 GB, past the whole serving
budget -- and it shares the *same* token cache, so a comparison measures the
ranking function rather than the tokeniser.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

import numpy as np
import scipy.sparse as sp

from medsearch._typing import FloatArray
from medsearch.logging_conf import get_logger
from medsearch.search.baseline import BaselineHit

logger = get_logger(__name__)

#: Term-frequency saturation. Higher means tf keeps mattering for longer;
#: 1.2 is the TREC default and the value nearly all published comparisons use.
DEFAULT_K1 = 1.2

#: Length-normalisation strength, 0 (none) to 1 (full). 0.75 is the default.
DEFAULT_B = 0.75


class BM25Baseline:
    """Okapi BM25 over a preprocessed token cache.

    Args:
        documents: Preprocessed token lists, in corpus row order.
        k1: Term-frequency saturation point.
        b: Length-normalisation strength, in ``[0, 1]``.

    Example:
        >>> bm25 = BM25Baseline([["lung", "failure"], ["vaccine"]])
        >>> bm25.search(["lung"], top_n=1)[0].row_id
        0
    """

    def __init__(
        self,
        documents: Iterable[Sequence[str]],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        docs = [list(d) for d in documents]
        self._n_documents = len(docs)
        self._k1 = k1
        self._b = b

        vocabulary: dict[str, int] = {}
        for tokens in docs:
            for token in tokens:
                if token not in vocabulary:
                    vocabulary[token] = len(vocabulary)
        self._vocabulary = vocabulary

        lengths = np.array([len(d) for d in docs], dtype=np.float32)
        self._avg_length = float(lengths.mean()) if len(lengths) else 0.0

        if not vocabulary or not docs:
            self._matrix = sp.csr_matrix((self._n_documents, 0), dtype=np.float32)
            self._idf = np.zeros(0, dtype=np.float32)
            logger.warning("BM25 baseline built over an empty vocabulary")
            return

        # COO triplets, never dense. Each stored value is the *fully weighted*
        # BM25 term contribution, so query time is one sparse column gather
        # rather than a re-weighting pass over the corpus.
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        document_frequency = np.zeros(len(vocabulary), dtype=np.float32)

        for row, tokens in enumerate(docs):
            if not tokens:
                continue
            counts = Counter(tokens)
            # Length normalisation for this document, shared by all its terms.
            norm = k1 * (1.0 - b + b * (len(tokens) / (self._avg_length or 1.0)))
            for token, tf in counts.items():
                column = vocabulary[token]
                document_frequency[column] += 1.0
                rows.append(row)
                cols.append(column)
                values.append(tf * (k1 + 1.0) / (tf + norm))

        # Robertson/Sparck-Jones IDF with the +0.5 smoothing, floored at zero.
        # Without the floor, a term appearing in more than half the corpus gets
        # a negative weight and actively demotes documents that contain it --
        # which on a COVID-only corpus would happen to the word "covid".
        idf = np.log(
            1.0 + (self._n_documents - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        self._idf = idf.astype(np.float32)

        self._matrix = sp.csr_matrix(
            (np.asarray(values, dtype=np.float32), (rows, cols)),
            shape=(self._n_documents, len(vocabulary)),
            dtype=np.float32,
        )
        logger.info(
            "BM25 baseline: %d documents x %d terms, %.1f MB sparse (k1=%.2f, b=%.2f)",
            self._n_documents,
            len(vocabulary),
            self._matrix.data.nbytes / 1024**2,
            k1,
            b,
        )

    @property
    def size(self) -> int:
        """Number of indexed documents."""
        return self._n_documents

    def score(self, tokens: Sequence[str]) -> FloatArray:
        """BM25 score for every document against a preprocessed query.

        Args:
            tokens: Query tokens, produced by the *same* preprocessor used on
                the corpus.

        Returns:
            ``(n_documents,) float32``. All zeros when no query term is in the
            vocabulary -- the caller decides what that means.
        """
        scores = np.zeros(self._n_documents, dtype=np.float32)
        if self._matrix.shape[1] == 0:
            return scores

        # A repeated query term contributes its weight once per occurrence,
        # which is the standard treatment.
        for token in tokens:
            column = self._vocabulary.get(token)
            if column is None:
                continue
            scores += (
                self._idf[column]
                * np.asarray(self._matrix[:, column].todense(), dtype=np.float32).ravel()
            )
        return scores

    def search(self, tokens: Sequence[str], *, top_n: int = 10) -> list[BaselineHit]:
        """Rank documents against a preprocessed query.

        Args:
            tokens: Query tokens from the shared preprocessor.
            top_n: Maximum hits to return.

        Returns:
            Hits ordered by descending score. Empty when nothing matches, so a
            no-match is distinguishable from a weak match rather than being
            padded with zero-scoring documents.
        """
        if not tokens or self._n_documents == 0:
            return []

        scores = self.score(tokens)
        if not scores.any():
            return []

        limit = min(top_n, self._n_documents)
        top = np.argpartition(-scores, limit - 1)[:limit]
        top = top[np.argsort(-scores[top])]
        return [BaselineHit(row_id=int(i), score=float(scores[i])) for i in top if scores[i] > 0.0]
