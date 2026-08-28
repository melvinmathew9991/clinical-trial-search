"""SIF weighting for document vectors.

Implements Arora, Liang & Ma (2017), *A Simple but Tough-to-Beat Baseline for
Sentence Embeddings*. Two steps, and the second is the one most
implementations omit:

1. **Smooth inverse frequency.** Weight each word vector by ``a / (a + p(w))``
   before averaging, so a rare, discriminating term dominates a common one.
2. **Common component removal.** Project the resulting document vectors onto
   their first principal component and subtract it. That direction encodes
   what every document in the corpus shares -- syntax, boilerplate, the fact
   that every abstract here is about COVID-19 -- and removing it is where much
   of SIF's benefit comes from.

Why this exists: the n=97 evaluation showed a TF-IDF baseline beating both
mean-pooled embedding models by 0.163 Recall@10 (p = 0.0003). The diagnosis was
that mean pooling treats ``colchicine`` and ``patients`` as equally
informative, discarding exactly the signal IDF supplies (PRD section 8.1). SIF
puts a frequency-derived weight back.

The principal component is a property of the *corpus*, so it is computed at
index-build time and persisted with the index. A query must have the same
component removed, or it is compared against documents living in a different
subspace.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from medsearch._typing import FloatArray
from medsearch.exceptions import EmptyCorpusError
from medsearch.logging_conf import get_logger

logger = get_logger(__name__)

#: Smoothing constant from the paper. Performance is famously flat over
#: 1e-3 to 1e-4; 1e-3 is the published default.
DEFAULT_SIF_A = 1e-3

_WEIGHTS_FILENAME = "sif_weights.json"


@dataclass(frozen=True, slots=True)
class SifWeights:
    """Per-token smooth-inverse-frequency weights derived from a corpus.

    Attributes:
        weights: Token -> ``a / (a + p(w))``.
        a: Smoothing constant used to build them.
        default: Weight for a token absent from the corpus. An unseen token has
            ``p(w) ~ 0``, so its weight approaches 1.0 -- maximally
            discriminating, which is the correct treatment for a rare term.
        total_tokens: Corpus size the frequencies came from, for provenance.
    """

    weights: dict[str, float]
    a: float
    default: float
    total_tokens: int

    @classmethod
    def from_corpus(
        cls, documents: Iterable[Sequence[str]], *, a: float = DEFAULT_SIF_A
    ) -> SifWeights:
        """Compute weights from a token corpus.

        Args:
            documents: Re-iterable of token lists, e.g. a
                :class:`~medsearch.preprocessing.pipeline.TokenCache`.
            a: Smoothing constant.

        Returns:
            Weights for every token observed.

        Raises:
            ValueError: The corpus produced no tokens.
        """
        counts: Counter[str] = Counter()
        for tokens in documents:
            counts.update(tokens)

        total = sum(counts.values())
        if total == 0:
            # EmptyCorpusError, not ValueError: "loaded, but no usable text" is
            # precisely what the typed class exists for (Rules section 4).
            raise EmptyCorpusError(
                "Cannot compute SIF weights: the corpus contains no tokens.\n"
                "  Expected: a token cache holding at least one term.\n"
                "  Fix: re-run `medsearch preprocess --force`. An empty cache "
                "usually means the text field was blank for every row."
            )

        weights = {token: a / (a + count / total) for token, count in counts.items()}

        logger.info(
            "SIF weights over %d tokens (%d distinct), a=%.0e | "
            "rarest weight %.4f, most common weight %.4f",
            total,
            len(weights),
            a,
            max(weights.values()),
            min(weights.values()),
        )
        return cls(weights=weights, a=a, default=1.0, total_tokens=total)

    def weight(self, token: str) -> float:
        """Weight for one token; ``default`` when unseen."""
        return self.weights.get(token, self.default)

    def weights_for(self, tokens: Sequence[str]) -> FloatArray:
        """Weights for a token sequence, as a ``float32`` array."""
        return np.asarray([self.weight(t) for t in tokens], dtype=np.float32)

    def save(self, directory: Path) -> None:
        """Persist alongside a model."""
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "a": self.a,
            "default": self.default,
            "total_tokens": self.total_tokens,
            "weights": self.weights,
        }
        (directory / _WEIGHTS_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
        logger.info("Saved SIF weights for %d tokens", len(self.weights))

    @classmethod
    def load(cls, directory: Path) -> SifWeights:
        """Read back weights saved by :meth:`save`."""
        payload = json.loads((directory / _WEIGHTS_FILENAME).read_text(encoding="utf-8"))
        return cls(
            weights=payload["weights"],
            a=float(payload["a"]),
            default=float(payload["default"]),
            total_tokens=int(payload["total_tokens"]),
        )

    @classmethod
    def exists(cls, directory: Path) -> bool:
        """True when weights are present at ``directory``."""
        return (directory / _WEIGHTS_FILENAME).exists()


def principal_component(matrix: FloatArray) -> FloatArray:
    """First right singular vector of a document matrix.

    Computed with a truncated SVD over the mean-centred matrix. At
    10,666 x 100 this is a few milliseconds and well inside the memory budget,
    so no randomised solver is needed.

    Args:
        matrix: Shape ``(n_documents, dim)``.

    Returns:
        Shape ``(dim,)`` unit vector -- the direction every document shares.
    """
    if matrix.shape[0] == 0:
        return np.zeros(matrix.shape[1], dtype=np.float32)

    centred = matrix - matrix.mean(axis=0, keepdims=True)
    # full_matrices=False keeps the factorisation at (n, dim) rather than
    # (n, n), which matters at 10k documents.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    component: FloatArray = np.asarray(vt[0], dtype=np.float32)
    return component


def remove_component(matrix: FloatArray, component: FloatArray) -> FloatArray:
    """Subtract the projection onto ``component`` from every row.

    ``v <- v - u (u^T v)``. Applied to documents at index-build time and to the
    query at search time; skipping it on either side compares vectors living in
    different subspaces.

    Args:
        matrix: Shape ``(n, dim)`` or ``(dim,)``.
        component: Shape ``(dim,)`` unit vector.

    Returns:
        Same shape as ``matrix``, ``float32``.
    """
    if component.size == 0 or not np.any(component):
        return matrix

    if matrix.ndim == 1:
        projected: FloatArray = (matrix - component * float(matrix @ component)).astype(
            np.float32, copy=False
        )
        return projected

    result: FloatArray = (matrix - np.outer(matrix @ component, component)).astype(
        np.float32, copy=False
    )
    return result
