"""Document index persistence.

The legacy project stored 10,666 x 100 floats as a 21 MB CSV and re-parsed
every float on each process start. The same data as ``float32`` ``.npy`` is
4.3 MB and memory-maps in milliseconds (ADR-002).

Rows are L2-normalised at build time, which is what makes ranking a single
matrix-vector product rather than a Python loop (ADR-003).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from medsearch.exceptions import ArtefactMismatchError, IndexBuildError
from medsearch.logging_conf import get_logger

logger = get_logger(__name__)

_VECTORS_FILENAME = "vectors.npy"
_ROW_IDS_FILENAME = "row_ids.npy"
_MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class DocumentIndex:
    """L2-normalised document vectors plus the manifest that identifies them.

    Attributes:
        vectors: Shape ``(n_documents, dim)`` ``float32``, rows unit-length.
        row_ids: Shape ``(n_documents,)`` ``int64``, positional ids into the
            corpus frame.
        model_fingerprint: Fingerprint of the model that produced these
            vectors. Checked on load so an index can never be silently paired
            with the wrong model -- the failure mode the legacy code shipped.
        model_kind: ``"skipgram"`` or ``"fasttext"``.
        field: ``"abstract"`` or ``"title"``.
        corpus_fingerprint: Digest of the corpus the vectors describe.
    """

    vectors: np.ndarray
    row_ids: np.ndarray
    model_fingerprint: str
    model_kind: str
    field: str
    corpus_fingerprint: str

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise IndexBuildError(
                f"Index vectors must be 2-D (n_documents, dim), got shape {self.vectors.shape}."
            )
        if len(self.row_ids) != len(self.vectors):
            raise IndexBuildError(
                f"row_ids length ({len(self.row_ids)}) does not match "
                f"vector count ({len(self.vectors)})."
            )

    @property
    def size(self) -> int:
        """Number of indexed documents."""
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return int(self.vectors.shape[1])

    @property
    def nbytes(self) -> int:
        """In-memory size of the vector matrix."""
        return int(self.vectors.nbytes)

    def save(self, directory: Path) -> None:
        """Write vectors, row ids, and manifest to ``directory``."""
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / _VECTORS_FILENAME, self.vectors.astype(np.float32, copy=False))
        np.save(directory / _ROW_IDS_FILENAME, self.row_ids.astype(np.int64, copy=False))

        manifest = {
            "model_fingerprint": self.model_fingerprint,
            "model_kind": self.model_kind,
            "field": self.field,
            "corpus_fingerprint": self.corpus_fingerprint,
            "documents": self.size,
            "dim": self.dim,
            "dtype": "float32",
            "normalized": True,
        }
        (directory / _MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        logger.info(
            "Saved index %s: %d docs x %d dims (%.1f MB)",
            directory.name,
            self.size,
            self.dim,
            self.nbytes / 1024**2,
        )

    @classmethod
    def load(
        cls,
        directory: Path,
        *,
        expected_fingerprint: str | None = None,
        mmap: bool = True,
    ) -> DocumentIndex:
        """Load an index, optionally verifying it against a model.

        Args:
            directory: Directory written by :meth:`save`.
            expected_fingerprint: If given, the manifest must match it.
            mmap: Memory-map the vectors. The matrix is read-only during
                search, so mapping keeps it out of RSS.

        Raises:
            IndexBuildError: The index is absent or incomplete.
            ArtefactMismatchError: The index was built by a different model.
        """
        manifest_file = directory / _MANIFEST_FILENAME
        if not manifest_file.exists():
            raise IndexBuildError(
                f"No index at {directory}.\n"
                f"  Fix: run `medsearch index build`."
            )

        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        actual = str(manifest["model_fingerprint"])
        if expected_fingerprint is not None and actual != expected_fingerprint:
            raise ArtefactMismatchError(expected=expected_fingerprint, actual=actual)

        mode = "r" if mmap else None
        vectors = np.load(directory / _VECTORS_FILENAME, mmap_mode=mode)
        row_ids = np.load(directory / _ROW_IDS_FILENAME)

        return cls(
            vectors=vectors,
            row_ids=row_ids,
            model_fingerprint=actual,
            model_kind=str(manifest["model_kind"]),
            field=str(manifest["field"]),
            corpus_fingerprint=str(manifest.get("corpus_fingerprint", "")),
        )

    @classmethod
    def exists(cls, directory: Path) -> bool:
        """True when a complete index is present at ``directory``."""
        return all(
            (directory / name).exists()
            for name in (_VECTORS_FILENAME, _ROW_IDS_FILENAME, _MANIFEST_FILENAME)
        )

    def close(self) -> None:
        """Release the memory-mapped file handle.

        Matters on Windows, which takes a mandatory lock on a mapped file: an
        index left mapped by a running Streamlit session cannot be overwritten
        by ``medsearch index build``, which fails with ``WinError 32`` --
        "the process cannot access the file because it is being used by
        another process".

        Safe to call on a non-mapped index, and safe to call twice.
        """
        base = getattr(self.vectors, "_mmap", None)
        if base is not None:  # numpy.memmap
            base.close()

    def __enter__(self) -> DocumentIndex:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
