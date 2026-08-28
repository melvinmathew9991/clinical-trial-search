"""Shared fixtures.

Tests never touch the network, the 29 MB corpus, or a trained model
(Rules.md section 5). Everything here is small enough that the whole suite
runs in seconds on the target laptop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from medsearch.config import Settings

# ---------------------------------------------------------------- corpus

RAW_CSV = """Date added,Trial ID,Title,Brief title,Abstract,Publication date,Phase
2021-06-04,NCT001,Coronavirus anxiety and emotional eating,Short A,"Patients with severe acute respiratory distress syndrome required ventilation.",2021-06-01,Phase 2
2021-06-05,NCT002,Lung failure in critical COVID-19 cases,Short B,"Acute lung injury and respiratory failure were observed in ventilated patients.",2021-06-02,Phase 3
2021-06-06,NCT003,Vaccine immunogenicity trial,Short C,"The vaccine produced antibody seroconversion in healthy adult volunteers.",2021-06-03,Phase 1
2021-06-07,NCT004,Kidney injury among hospitalised patients,Short D,"Renal impairment and kidney injury occurred in hospitalised coronavirus patients.",2021-06-04,Phase 2
2021-06-08,NCT005,Breathing difficulty assessment,Short E,"Breathing difficulty and shortness of breath were assessed in outpatients.",2021-06-05,Phase 1
"""


@pytest.fixture
def corpus_csv(tmp_path: Path) -> Path:
    """A five-row corpus with the real column names."""
    path = tmp_path / "dimension-covid.csv"
    path.write_text(RAW_CSV, encoding="utf-8")
    return path


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
def settings(tmp_path: Path) -> Settings:
    """Settings pointing entirely at ``tmp_path``."""
    return Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        report_dir=tmp_path / "reports",
        vector_size=8,
        window=2,
        min_count=1,
        epochs=1,
        workers=1,
        fasttext_bucket=1_000,
    )


# ---------------------------------------------------------------- vectors


class FakeKeyedVectors:
    """Minimal stand-in for gensim ``KeyedVectors``.

    Lets the embedding and search layers be tested without gensim installed
    and without a training run.
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
