"""Shared type aliases and structural protocols.

Foundation-layer module: numpy and stdlib only, no ``medsearch`` imports.

Exists because the first pass typed gensim objects and numpy arrays as bare
``object``, which silenced nothing and cost ``mypy --strict`` 21 errors. A
structural protocol gives the same "no hard dependency on gensim's classes"
property while keeping the calls type-checked.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

#: A ``float32`` array of any shape. Document vectors, index matrices, and
#: similarity scores are all this dtype -- ``float64`` doubles memory for no
#: retrieval benefit (Rules.md section 1).
FloatArray = np.ndarray[Any, np.dtype[np.float32]]

#: An ``int64`` array. Used for corpus row ids.
IntArray = np.ndarray[Any, np.dtype[np.int64]]


@runtime_checkable
class WordVectors(Protocol):
    """Structural type for gensim's ``KeyedVectors``.

    Only the members this project actually uses are declared. Typing against
    the protocol rather than importing ``gensim.models.KeyedVectors`` keeps
    the embedding layer testable with a lightweight fake (see
    ``tests/conftest.py::FakeKeyedVectors``) and avoids a hard dependency in
    modules that only read vectors.
    """

    #: Embedding dimensionality.
    vector_size: int

    #: Vocabulary in index order.
    index_to_key: list[str]

    #: The vector matrix itself, ``(vocabulary, vector_size) float32``.
    #: Declared because provenance needs to checksum the artefact, not just
    #: the configuration that produced it -- two runs of one config hold
    #: different vectors whenever gensim uses more than one worker.
    vectors: FloatArray

    def __contains__(self, key: str) -> bool:
        """True when a vector can be produced for ``key``.

        For FastText this is True for unseen words too, since a vector can be
        synthesised from character n-grams.
        """
        ...

    def __len__(self) -> int:
        """Vocabulary size."""
        ...

    def __getitem__(self, key: Any) -> Any:
        """Vector for one key, or a stacked array for a list of keys."""
        ...
