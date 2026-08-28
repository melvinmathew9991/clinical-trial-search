"""Exception hierarchy.

Every error raised inside ``medsearch`` derives from :class:`MedSearchError`,
so user-facing surfaces (CLI, Streamlit) can catch exactly one type and render
``str(exc)`` without leaking a traceback (Rules.md section 4).

Messages follow a fixed shape: **what failed, what was expected, what to do.**
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


class MedSearchError(Exception):
    """Base class. Never raised directly."""


# ---------------------------------------------------------------- config
class ConfigurationError(MedSearchError):
    """A setting is missing, malformed, or mutually inconsistent."""


# ---------------------------------------------------------------- data
class DataError(MedSearchError):
    """Base for corpus loading and validation failures."""


class CorpusNotFoundError(DataError):
    """The corpus file does not exist at the configured path."""

    def __init__(self, path: Path) -> None:
        super().__init__(
            f"Corpus not found at {path}.\n"
            f"  Expected: the Dimensions COVID-19 clinical-trial CSV export.\n"
            f"  Fix: place the Dimensions export at {path}.\n"
            f"  Note: the legacy Part_1 copy was deleted once the migration was "
            f"verified, so `make data` can no longer recover it."
        )
        self.path = path


class SchemaValidationError(DataError):
    """The corpus is missing one or more required columns."""

    def __init__(self, missing: Sequence[str], found: Sequence[str]) -> None:
        preview = list(found)[:8]
        suffix = ", ..." if len(found) > 8 else ""
        super().__init__(
            f"Corpus is missing required column(s): {sorted(missing)}.\n"
            f"  Found: {preview}{suffix}\n"
            f"  Fix: check that the CSV is the full Dimensions export and not a "
            f"filtered or re-exported subset."
        )
        self.missing = list(missing)
        self.found = list(found)


class EmptyCorpusError(DataError):
    """The corpus loaded successfully but contains no usable rows."""


# ---------------------------------------------------------------- model
class ModelError(MedSearchError):
    """Base for embedding-model failures."""


class ModelNotTrainedError(ModelError):
    """A model artefact was requested before it was trained."""

    def __init__(self, model: str, path: Path) -> None:
        super().__init__(
            f"No trained '{model}' model at {path}.\n"
            f"  Fix: run `medsearch train --model {model}` "
            f"(add `--limit 2000` for a fast low-memory run)."
        )
        self.model = model
        self.path = path


class ArtefactMismatchError(ModelError):
    """An index was built by a different model than the one now loaded.

    Guards against the legacy failure mode where FastText result files were
    silently populated with Skip-gram vectors (reference/legacy/modular-code ``engine.py`` lines
    36 and 41).
    """

    def __init__(self, expected: str, actual: str, subject: str = "fingerprint") -> None:
        super().__init__(
            f"Index/model {subject} mismatch.\n"
            f"  Index was built from: {actual}\n"
            f"  Model now loaded:     {expected}\n"
            f"  Fix: rebuild the index with `medsearch index build`."
        )
        self.expected = expected
        self.actual = actual


class StaleIndexError(ModelError):
    """The corpus changed after the index was built.

    Distinct from :class:`ArtefactMismatchError`, which is about the *model*.
    This one is more dangerous: row ids are positional, so a stale index still
    resolves — to the wrong documents. A search result would carry one trial's
    title beside another trial's relevance score, and nothing would look
    broken.

    Realistic in production: the Azure pipeline is triggered by a new CSV
    landing in blob storage. If training fails after the drop but the app
    restarts, it would serve the previous corpus's index against the new data.
    """

    def __init__(self, expected: str, actual: str, *, documents: int) -> None:
        super().__init__(
            f"Index is stale: the corpus changed after it was built.\n"
            f"  Index was built from corpus: {actual}\n"
            f"  data/raw currently holds:    {expected}\n"
            f"  Why this matters: row ids are positional, so this index would\n"
            f"  resolve to the WRONG documents rather than simply failing.\n"
            f"  ({documents:,} vectors affected.)\n"
            f"  Fix: medsearch train && medsearch index build"
        )
        self.expected = expected
        self.actual = actual


class IndexBuildError(MedSearchError):
    """The document index could not be built or loaded."""


# ---------------------------------------------------------------- resources
class ResourceError(MedSearchError):
    """Insufficient memory or disk to safely start a stage.

    Raised *before* an expensive operation begins, never part-way through
    (Rules.md section 4).
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
