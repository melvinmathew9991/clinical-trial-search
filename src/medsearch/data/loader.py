"""Corpus loading.

Replaces the legacy ``utils.read_data``, which read all 21 columns, built a
second identical frame it never used, and silently returned ``df.iloc[:100, :]``
-- so every model the legacy project shipped was trained on 100 of 10,666
abstracts (ADR-007).

Here: only required columns are read, the full corpus is returned by default,
and any row cap is explicit and logged.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import pandas as pd

from medsearch.data.schema import CorpusSchema
from medsearch.exceptions import CorpusNotFoundError, EmptyCorpusError, SchemaValidationError
from medsearch.logging_conf import get_logger

logger = get_logger(__name__)

_HASH_CHUNK_BYTES = 1 << 20  # 1 MB -- streams the hash instead of reading 29 MB into RAM
_FINGERPRINT_LENGTH = 16


def corpus_fingerprint(path: Path) -> str:
    """Return a short stable digest of the corpus file's bytes.

    Used to key the preprocessing cache and to stamp trained models, so an
    index can prove which corpus and model produced it.

    Args:
        path: Corpus CSV.

    Returns:
        First 16 hex characters of the SHA-256 digest.

    Raises:
        CorpusNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise CorpusNotFoundError(path)

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()[:_FINGERPRINT_LENGTH]


def load_corpus(path: Path, *, limit: int | None = None) -> pd.DataFrame:
    """Load and validate the clinical-trial corpus.

    Reads only the six columns in :class:`~medsearch.data.schema.CorpusSchema`
    and renames them to snake_case.

    Args:
        path: Corpus CSV.
        limit: Optional row cap for fast local iteration. **Defaults to
            ``None``, meaning the full corpus.** When set, a warning is logged
            -- a sampled corpus must never be mistaken for a production run.

    Returns:
        DataFrame with canonical column names and a reset integer index. The
        index position is the document's row id, used throughout the index.

    Raises:
        CorpusNotFoundError: The file does not exist.
        SchemaValidationError: A required column is absent.
        EmptyCorpusError: No usable rows remain after cleaning.
    """
    if not path.exists():
        raise CorpusNotFoundError(path)

    header = pd.read_csv(path, nrows=0)
    missing = CorpusSchema.missing_required(header.columns)
    if missing:
        raise SchemaValidationError(missing=sorted(missing), found=[str(c) for c in header.columns])

    # Only read columns that are actually present, so an export missing an
    # optional column (Phase, Date added) still loads.
    usecols = [c for c in CorpusSchema.source_columns() if c in header.columns]
    dtypes = {k: v for k, v in CorpusSchema.dtypes.items() if k in usecols}

    # The corpus contains newlines embedded inside quoted abstracts; the C
    # parser handles these correctly with the default quoting rules.
    # cast on dtype only: pandas-stubs types the mapping more narrowly than
    # the runtime accepts. The call itself is correct.
    frame: pd.DataFrame = pd.read_csv(
        path,
        usecols=usecols,
        dtype=cast(Any, dtypes),
        nrows=limit,
        engine="c",
    )
    frame = frame.rename(columns=CorpusSchema.column_map)

    # Drop rows with no text in either field -- they cannot be embedded and
    # would otherwise become zero vectors that pollute similarity ranking.
    text_cols = [c for c in CorpusSchema.text_fields if c in frame.columns]
    frame = frame.dropna(subset=text_cols, how="all").reset_index(drop=True)

    if frame.empty:
        raise EmptyCorpusError(f"Corpus at {path} has no rows with usable text in {text_cols}.")

    if limit is not None:
        logger.warning(
            "SAMPLED CORPUS: loaded %d rows because limit=%d was set. "
            "Artefacts built from this run are for development only.",
            len(frame),
            limit,
        )
    else:
        logger.info("Loaded %d documents from %s", len(frame), path.name)

    return frame


def iter_text(frame: pd.DataFrame, field: str) -> list[str]:
    """Extract one text column as a list of strings, nulls coerced to ``""``.

    Args:
        frame: Corpus frame from :func:`load_corpus`.
        field: Canonical field name, e.g. ``"abstract"``.

    Returns:
        One string per row, aligned with the frame's positional index.

    Raises:
        SchemaValidationError: If ``field`` is not a column.
    """
    if field not in frame.columns:
        raise SchemaValidationError(missing=[field], found=list(frame.columns))
    values: list[str] = frame[field].fillna("").astype(str).tolist()
    return values
