"""SIF weighting, principal component, and weighted pooling.

SIF was implemented as the remediation for the TF-IDF gap and then measured to
*not* work (PRD §8.2). These tests exist so the negative result is reproducible
and so the code stays correct if anyone revisits it: an implementation bug
would be a far more embarrassing explanation for the null result than the real
one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from medsearch.embeddings.document import DocumentEmbedder
from medsearch.embeddings.weighting import (
    DEFAULT_SIF_A,
    SifWeights,
    principal_component,
    remove_component,
)


@pytest.fixture
def corpus() -> list[list[str]]:
    """`patient` in every document, `colchicine` in exactly one."""
    return [
        ["patient", "patient", "lung"],
        ["patient", "lung", "failure"],
        ["patient", "colchicine"],
    ]


class TestSifWeights:
    def test_rare_token_outweighs_common(self, corpus: list[list[str]]) -> None:
        # The whole point: `colchicine` (1 of 8) must dominate `patient` (4 of 8).
        w = SifWeights.from_corpus(corpus)
        assert w.weight("colchicine") > w.weight("patient")

    def test_weight_formula(self, corpus: list[list[str]]) -> None:
        w = SifWeights.from_corpus(corpus, a=1e-3)
        # 8 tokens across the three documents; `patient` appears 4 times.
        expected = 1e-3 / (1e-3 + 4 / 8)
        assert w.weight("patient") == pytest.approx(expected)

    def test_weights_are_bounded(self, corpus: list[list[str]]) -> None:
        w = SifWeights.from_corpus(corpus)
        assert all(0.0 < v <= 1.0 for v in w.weights.values())

    def test_unseen_token_gets_maximum_weight(self, corpus: list[list[str]]) -> None:
        # p(w) ~ 0 for an unseen token, so its weight approaches 1.0 -- the
        # correct treatment for a maximally rare term.
        w = SifWeights.from_corpus(corpus)
        assert w.weight("neverseen") == 1.0

    def test_smaller_a_sharpens_the_contrast(self, corpus: list[list[str]]) -> None:
        loose = SifWeights.from_corpus(corpus, a=1e-2)
        tight = SifWeights.from_corpus(corpus, a=1e-4)
        ratio = lambda w: w.weight("colchicine") / w.weight("patient")  # noqa: E731
        assert ratio(tight) > ratio(loose)

    def test_default_a_is_the_published_value(self) -> None:
        assert DEFAULT_SIF_A == 1e-3

    def test_empty_corpus_rejected(self) -> None:
        with pytest.raises(ValueError, match="no tokens"):
            SifWeights.from_corpus([[], []])

    def test_weights_for_returns_float32_array(self, corpus: list[list[str]]) -> None:
        w = SifWeights.from_corpus(corpus)
        arr = w.weights_for(["patient", "colchicine"])
        assert arr.dtype == np.float32
        assert arr.shape == (2,)

    def test_round_trip(self, corpus: list[list[str]], tmp_path: Path) -> None:
        original = SifWeights.from_corpus(corpus)
        original.save(tmp_path)
        assert SifWeights.exists(tmp_path)
        loaded = SifWeights.load(tmp_path)
        assert loaded.weight("colchicine") == pytest.approx(original.weight("colchicine"))
        assert loaded.a == original.a

    def test_absent_before_saving(self, tmp_path: Path) -> None:
        assert not SifWeights.exists(tmp_path)


class TestPrincipalComponent:
    def test_is_unit_length(self) -> None:
        rng = np.random.default_rng(0)
        matrix = rng.random((50, 8)).astype(np.float32)
        assert np.linalg.norm(principal_component(matrix)) == pytest.approx(1.0, abs=1e-5)

    def test_recovers_a_planted_direction(self) -> None:
        rng = np.random.default_rng(1)
        direction = np.zeros(6, dtype=np.float32)
        direction[2] = 1.0
        matrix = (
            rng.normal(0, 0.01, (200, 6)) + np.outer(rng.normal(0, 1, 200), direction)
        ).astype(np.float32)
        # Sign is arbitrary in an SVD, so compare absolute alignment.
        assert abs(float(principal_component(matrix) @ direction)) > 0.95

    def test_empty_matrix_is_safe(self) -> None:
        assert principal_component(np.zeros((0, 5), dtype=np.float32)).shape == (5,)


class TestRemoveComponent:
    def test_removed_direction_is_orthogonal_afterwards(self) -> None:
        rng = np.random.default_rng(2)
        matrix = rng.random((40, 6)).astype(np.float32)
        u = principal_component(matrix)
        assert np.allclose(remove_component(matrix, u) @ u, 0.0, atol=1e-5)

    def test_works_on_a_single_vector(self) -> None:
        u = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        assert np.allclose(remove_component(v, u), [0.0, 4.0, 0.0])

    def test_zero_component_is_a_no_op(self) -> None:
        matrix = np.ones((3, 4), dtype=np.float32)
        assert np.allclose(remove_component(matrix, np.zeros(4, dtype=np.float32)), matrix)

    def test_output_is_float32(self) -> None:
        matrix = np.ones((3, 4), dtype=np.float32)
        u = principal_component(np.random.default_rng(3).random((3, 4)).astype(np.float32))
        assert remove_component(matrix, u).dtype == np.float32


class TestWeightedPooling:
    def test_weights_change_the_result(self, fake_vectors: object, corpus: list[list[str]]) -> None:
        w = SifWeights.from_corpus([["lung", "lung", "lung", "kidney"]])
        plain = DocumentEmbedder(fake_vectors).embed(["lung", "kidney"])
        weighted = DocumentEmbedder(fake_vectors, weights=w).embed(["lung", "kidney"])
        assert not np.allclose(plain, weighted)

    def test_weighting_pulls_towards_the_rare_token(self, fake_vectors: object) -> None:
        # `lung` common, `kidney` rare -> the weighted vector must sit closer
        # to `kidney` than the plain mean does.
        w = SifWeights.from_corpus([["lung"] * 20 + ["kidney"]])
        kidney = np.asarray(fake_vectors["kidney"])  # type: ignore[index]
        plain = DocumentEmbedder(fake_vectors).embed(["lung", "kidney"])
        weighted = DocumentEmbedder(fake_vectors, weights=w).embed(["lung", "kidney"])
        unit = lambda v: v / np.linalg.norm(v)  # noqa: E731
        assert float(unit(weighted) @ unit(kidney)) > float(unit(plain) @ unit(kidney))

    def test_no_weights_is_plain_mean(self, fake_vectors: object) -> None:
        embedder = DocumentEmbedder(fake_vectors)
        expected = np.mean(
            [fake_vectors["lung"], fake_vectors["kidney"]],  # type: ignore[index]
            axis=0,
        )
        assert np.allclose(embedder.embed(["lung", "kidney"]), expected, atol=1e-6)

    def test_all_oov_still_returns_zero_vector(self, fake_vectors: object) -> None:
        w = SifWeights.from_corpus([["lung"]])
        vector = DocumentEmbedder(fake_vectors, weights=w).embed(["zzzz"])
        assert np.count_nonzero(vector) == 0
        assert not np.isnan(vector).any()

    def test_output_dtype_and_shape(self, fake_vectors: object) -> None:
        w = SifWeights.from_corpus([["lung", "kidney"]])
        vector = DocumentEmbedder(fake_vectors, weights=w).embed(["lung", "kidney"])
        assert vector.dtype == np.float32
        assert vector.shape == (4,)

    def test_single_token_matches_its_vector(self, fake_vectors: object) -> None:
        # A weighted average over one token is that token, whatever the weight.
        w = SifWeights.from_corpus([["lung", "lung", "kidney"]])
        embedder = DocumentEmbedder(fake_vectors, weights=w)
        assert np.allclose(embedder.embed(["lung"]), fake_vectors["lung"], atol=1e-6)  # type: ignore[index]
