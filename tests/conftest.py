"""Shared fixtures.

Tests never touch the network, the 29 MB corpus, or a trained production
model (Rules.md section 5). Everything here is small enough that the whole
suite runs in seconds on the target laptop.

The sample corpus is a committed file rather than an inline string so that
``Architecture.md`` section 3 and the repository agree, and so the fixture can
be inspected and edited like data rather than code.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from medsearch.config import Settings

FIXTURES = Path(__file__).parent / "fixtures"

#: The committed 20-row corpus. Documents cluster into six topics
#: (respiratory, vaccine, renal, coagulation, inflammation, antiviral) so
#: retrieval assertions can be meaningful rather than arbitrary.
SAMPLE_CORPUS = FIXTURES / "sample_corpus.csv"


@pytest.fixture
def corpus_csv(tmp_path: Path) -> Path:
    """A copy of the sample corpus inside ``tmp_path``.

    Copied rather than used in place so a test that mutates it cannot corrupt
    the committed fixture.
    """
    destination = tmp_path / "dimension-covid.csv"
    shutil.copy2(SAMPLE_CORPUS, destination)
    return destination


@pytest.fixture
def corpus_missing_abstract(tmp_path: Path) -> Path:
    """A corpus lacking the required ``Abstract`` column."""
    path = tmp_path / "broken.csv"
    path.write_text(
        "Date added,Trial ID,Title,Publication date\n2021-06-04,NCT001,A title,2021-06-01\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def empty_corpus(tmp_path: Path) -> Path:
    """A schema-valid corpus whose text columns are all empty."""
    path = tmp_path / "empty.csv"
    path.write_text(
        "Date added,Trial ID,Title,Abstract,Publication date\n2021-06-04,NCT001,,,2021-06-01\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def settings(tmp_path: Path, corpus_csv: Path) -> Settings:
    """Settings pointing entirely at ``tmp_path``.

    Deliberately minimal hyperparameters: 16 dimensions (the floor ``Settings``
    permits), one epoch, one worker, and a 1,000 bucket. Over 20 documents that
    trains in well under a second, which is what keeps the integration test
    inside the 30-second suite budget.

    ``workers=1`` also makes training deterministic for the reproducibility
    test -- with more threads, interleaving makes runs near-identical but not
    bit-exact.
    """
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True, exist_ok=True)
    shutil.copy2(corpus_csv, data_dir / "raw" / "dimension-covid.csv")

    return Settings(
        data_dir=data_dir,
        model_dir=tmp_path / "models",
        report_dir=tmp_path / "reports",
        vector_size=16,
        window=2,
        min_count=1,
        epochs=1,
        workers=1,
        seed=42,
        fasttext_bucket=1_000,
        min_free_memory_gb=0.0,
        warn_free_memory_gb=0.0,
    )


# ---------------------------------------------------------------- vectors


class FakeKeyedVectors:
    """Minimal stand-in for gensim ``KeyedVectors``.

    Satisfies :class:`medsearch._typing.WordVectors` structurally, so the
    embedding and search layers can be tested without gensim and without a
    training run.
    """

    def __init__(self, vocabulary: dict[str, np.ndarray], vector_size: int) -> None:
        self._vectors = vocabulary
        self.vector_size = vector_size
        self.index_to_key = list(vocabulary)
        self.bucket = 0

    def __contains__(self, key: str) -> bool:
        return key in self._vectors

    def __len__(self) -> int:
        return len(self._vectors)

    def __getitem__(self, key: str | list[str]) -> np.ndarray:
        if isinstance(key, list):
            return np.vstack([self._vectors[k] for k in key])
        return self._vectors[key]


@pytest.fixture
def fake_vectors() -> FakeKeyedVectors:
    """A 4-dimensional toy vocabulary with deliberate structure.

    ``lung``/``respiratory``/``breathing`` cluster together; ``vaccine`` and
    ``kidney`` point elsewhere. That makes similarity assertions meaningful
    rather than arbitrary.
    """
    rng = np.random.default_rng(0)

    def unit(v: list[float]) -> np.ndarray:
        arr = np.asarray(v, dtype=np.float32)
        return arr / np.linalg.norm(arr)

    vocabulary = {
        "lung": unit([1.0, 0.1, 0.0, 0.0]),
        "respiratory": unit([0.95, 0.2, 0.0, 0.0]),
        "breathing": unit([0.9, 0.3, 0.05, 0.0]),
        "failure": unit([0.8, 0.4, 0.1, 0.0]),
        "vaccine": unit([0.0, 0.0, 1.0, 0.2]),
        "antibody": unit([0.0, 0.05, 0.95, 0.25]),
        "kidney": unit([0.0, 1.0, 0.0, 0.3]),
        "renal": unit([0.05, 0.95, 0.0, 0.35]),
        "patient": unit(list(rng.random(4))),
    }
    return FakeKeyedVectors(vocabulary, vector_size=4)


@pytest.fixture
def fasttext_like_vectors(fake_vectors: FakeKeyedVectors) -> FakeKeyedVectors:
    """Toy vectors that advertise character n-grams, like FastText.

    Exercises the ``_has_ngrams`` branch of ``DocumentEmbedder``.
    """
    fake_vectors.bucket = 1_000
    return fake_vectors


class StubPreprocessor:
    """Whitespace tokenizer, so search tests need no NLTK data."""

    def transform(self, text: str) -> list[str]:
        return [t for t in text.lower().split() if t]


@pytest.fixture
def stub_preprocessor() -> StubPreprocessor:
    """A deterministic preprocessor with no external data dependency."""
    return StubPreprocessor()
