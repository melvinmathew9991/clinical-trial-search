"""Document embedding by mean pooling.

Replaces the legacy ``get_mean_vector``, whose single line

    words = [w for w in word_tokenize(words) if w in list(model.wv.index_to_key)]

rebuilt a ~30,000-element Python list **once per document** and then did a
linear scan through it per token. Across 10,666 documents that is on the order
of 3x10^8 list-element constructions -- the dominant cost of the entire legacy
pipeline (ADR-004).

Here the vocabulary becomes a ``frozenset`` once, in ``__init__``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from medsearch.logging_conf import get_logger

logger = get_logger(__name__)


class DocumentEmbedder:
    """Turn token lists into fixed-length document vectors.

    A document vector is the arithmetic mean of its in-vocabulary word
    vectors. Documents with no in-vocabulary token get a zero vector and are
    counted -- the caller decides what to do about them rather than silently
    inheriting a NaN in the cosine denominator, which is what the legacy code
    did (Rules.md section 4).

    Args:
        vectors: A gensim ``KeyedVectors``.

    Attributes:
        dim: Embedding dimensionality.
        oov_documents: Count of all-out-of-vocabulary documents seen so far.
    """

    def __init__(self, vectors: object) -> None:
        self._vectors = vectors
        self.dim: int = int(vectors.vector_size)  # type: ignore[attr-defined]

        # Built ONCE. This is the fix for ADR-004.
        # FastText can synthesise a vector for an unseen word from its
        # character n-grams, so membership is looked up through the model's
        # own __contains__ when available rather than a fixed word list.
        self._has_ngrams = bool(getattr(vectors, "bucket", 0))
        self._vocabulary: frozenset[str] = frozenset(
            vectors.index_to_key  # type: ignore[attr-defined]
        )
        self.oov_documents = 0

        logger.debug(
            "DocumentEmbedder ready: dim=%d vocabulary=%d ngrams=%s",
            self.dim,
            len(self._vocabulary),
            self._has_ngrams,
        )

    def _known(self, tokens: Sequence[str]) -> list[str]:
        """Filter tokens to those the model can vectorise. O(1) per token."""
        if self._has_ngrams:
            # FastText: an out-of-vocabulary word still has a vector derived
            # from its character n-grams, so keep anything the model accepts.
            return [t for t in tokens if t in self._vocabulary or self._can_infer(t)]
        return [t for t in tokens if t in self._vocabulary]

    def _can_infer(self, token: str) -> bool:
        try:
            return bool(self._vectors.__contains__(token))  # type: ignore[attr-defined]
        except (KeyError, AttributeError):  # pragma: no cover
            return False

    def embed(self, tokens: Sequence[str]) -> np.ndarray:
        """Embed one document.

        Args:
            tokens: Preprocessed tokens.

        Returns:
            Shape ``(dim,)`` ``float32``. All zeros when no token is
            vectorisable; ``oov_documents`` is incremented in that case.
        """
        known = self._known(tokens)
        if not known:
            self.oov_documents += 1
            return np.zeros(self.dim, dtype=np.float32)

        # KeyedVectors.__getitem__ with a list returns a stacked (n, dim)
        # array in one call -- far cheaper than a Python accumulation loop.
        matrix = np.asarray(self._vectors[known], dtype=np.float32)  # type: ignore[index]
        return matrix.mean(axis=0, dtype=np.float32)

    def embed_corpus(
        self,
        documents: Iterable[Sequence[str]],
        *,
        chunk_size: int = 1_000,
        total: int | None = None,
    ) -> np.ndarray:
        """Embed a whole corpus, in chunks.

        Chunking bounds peak memory: instead of one giant Python list of
        arrays, results are consolidated into ``float32`` blocks every
        ``chunk_size`` documents.

        Args:
            documents: Iterable of token lists.
            chunk_size: Documents per consolidation.
            total: Expected document count, for progress logging only.

        Returns:
            Shape ``(n_documents, dim)`` ``float32``.
        """
        self.oov_documents = 0
        blocks: list[np.ndarray] = []
        buffer: list[np.ndarray] = []
        seen = 0

        for tokens in documents:
            buffer.append(self.embed(tokens))
            seen += 1
            if len(buffer) >= chunk_size:
                blocks.append(np.vstack(buffer))
                buffer.clear()
                if total:
                    logger.info("Embedded %d/%d documents", seen, total)
                else:
                    logger.debug("Embedded %d documents", seen)

        if buffer:
            blocks.append(np.vstack(buffer))

        if not blocks:
            return np.zeros((0, self.dim), dtype=np.float32)

        matrix = np.vstack(blocks).astype(np.float32, copy=False)

        if self.oov_documents:
            logger.warning(
                "%d/%d documents (%.1f%%) had no in-vocabulary tokens and are zero vectors. "
                "They will never be returned as search results.",
                self.oov_documents,
                seen,
                100.0 * self.oov_documents / max(seen, 1),
            )
        return matrix


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row, leaving zero rows as zero.

    Pre-normalising the index is what reduces cosine similarity to a single
    matrix-vector product at query time (ADR-003). Zero rows are left alone
    rather than producing NaN, so an all-OOV document simply scores 0 against
    every query.

    Args:
        matrix: Shape ``(n, dim)``.

    Returns:
        Shape ``(n, dim)`` ``float32``, each non-zero row of unit length.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Guard the divide: zero-norm rows stay zero instead of becoming NaN.
    np.maximum(norms, np.finfo(np.float32).tiny, out=norms)
    return (matrix / norms).astype(np.float32, copy=False)
