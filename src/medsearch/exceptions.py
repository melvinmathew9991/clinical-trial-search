"""Exception hierarchy.

Every error raised inside ``medsearch`` derives from :class:`MedSearchError`,
so user-facing surfaces (CLI, Streamlit) can catch exactly one type and render
``str(exc)`` without leaking a traceback (Rules.md section 4).

Messages follow a fixed shape: **what failed, what was expected, what to do.**
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


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
            f"  Fix: run `make data` to migrate it from the legacy Part_1 folder, "
            f"or place the file at {path} yourself."
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
    silently populated with Skip-gram vectors (Part 1 ``engine.py`` lines
    36 and 41).
    """

    def __init__(self, expected: str, actual: str) -> None:
        super().__init__(
            f"Index/model fingerprint mismatch.\n"
            f"  Index was built from model fingerprint: {actual}\n"
            f"  Currently loaded model fingerprint:     {expected}\n"
            f"  Fix: rebuild the index with `medsearch index build`."
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
