"""DocumentEmbedder mean pooling and L2 normalisation."""

from __future__ import annotations

import numpy as np
import pytest

from medsearch.embeddings.document import DocumentEmbedder, l2_normalize


class TestEmbed:
    def test_shape_and_dtype(self, fake_vectors: object) -> None:
        vector = DocumentEmbedder(fake_vectors).embed(["lung", "failure"])
        assert vector.shape == (4,)
        assert vector.dtype == np.float32

    def test_is_the_mean_of_member_vectors(self, fake_vectors: object) -> None:
        embedder = DocumentEmbedder(fake_vectors)
        expected = np.mean([fake_vectors["lung"], fake_vectors["renal"]], axis=0)  # type: ignore[index]
        assert np.allclose(embedder.embed(["lung", "renal"]), expected, atol=1e-6)

    def test_single_token_returns_that_vector(self, fake_vectors: object) -> None:
        embedder = DocumentEmbedder(fake_vectors)
        assert np.allclose(embedder.embed(["lung"]), fake_vectors["lung"], atol=1e-6)  # type: ignore[index]

    def test_unknown_tokens_are_ignored(self, fake_vectors: object) -> None:
        embedder = DocumentEmbedder(fake_vectors)
        assert np.allclose(embedder.embed(["lung", "zzzz"]), embedder.embed(["lung"]))

    def test_token_order_does_not_matter(self, fake_vectors: object) -> None:
        embedder = DocumentEmbedder(fake_vectors)
        assert np.allclose(embedder.embed(["lung", "renal"]), embedder.embed(["renal", "lung"]))

    def test_all_oov_gives_zero_vector(self, fake_vectors: object) -> None:
        vector = DocumentEmbedder(fake_vectors).embed(["zzzz", "qqqq"])
        assert np.count_nonzero(vector) == 0
        assert not np.isnan(vector).any()

    def test_empty_tokens_give_zero_vector(self, fake_vectors: object) -> None:
        assert np.count_nonzero(DocumentEmbedder(fake_vectors).embed([])) == 0

    def test_oov_documents_are_counted(self, fake_vectors: object) -> None:
        embedder = DocumentEmbedder(fake_vectors)
        embedder.embed(["zzzz"])
        embedder.embed(["qqqq"])
        embedder.embed(["lung"])
        assert embedder.oov_documents == 2

    def test_dim_matches_the_model(self, fake_vectors: object) -> None:
        assert DocumentEmbedder(fake_vectors).dim == 4


class TestVocabularyIsBuiltOnce:
    """ADR-004: the legacy code rebuilt a ~30k list per document."""

    def test_index_to_key_is_read_once_at_construction(self, fake_vectors: object) -> None:
        reads = {"count": 0}
        real = fake_vectors.index_to_key  # type: ignore[attr-defined]

        class Counting:
            vector_size = 4
            bucket = 0

            @property
            def index_to_key(self) -> list[str]:
                reads["count"] += 1
                return real

            def __contains__(self, key: str) -> bool:
                return key in fake_vectors  # type: ignore[operator]

            def __len__(self) -> int:
                return len(real)

            def __getitem__(self, key: object) -> object:
                return fake_vectors[key]  # type: ignore[index]

        embedder = DocumentEmbedder(Counting())
        after_construction = reads["count"]
        for _ in range(50):
            embedder.embed(["lung", "failure"])
        assert reads["count"] == after_construction


class TestFastTextInference:
    def test_ngram_model_may_infer_unseen_words(self, fasttext_like_vectors: object) -> None:
        # A FastText-like model advertises `bucket`, so membership goes
        # through __contains__ rather than the fixed word list.
        embedder = DocumentEmbedder(fasttext_like_vectors)
        assert embedder.embed(["lung"]).shape == (4,)

    def test_still_zero_when_nothing_is_inferable(self, fasttext_like_vectors: object) -> None:
        embedder = DocumentEmbedder(fasttext_like_vectors)
        assert np.count_nonzero(embedder.embed(["zzzz"])) == 0


class TestEmbedCorpus:
    def test_shape(self, fake_vectors: object) -> None:
        matrix = DocumentEmbedder(fake_vectors).embed_corpus([["lung"], ["renal"], ["vaccine"]])
        assert matrix.shape == (3, 4)

    def test_dtype_is_float32(self, fake_vectors: object) -> None:
        # float64 would double index memory for no retrieval benefit.
        matrix = DocumentEmbedder(fake_vectors).embed_corpus([["lung"]])
        assert matrix.dtype == np.float32

    def test_empty_corpus_gives_empty_matrix(self, fake_vectors: object) -> None:
        matrix = DocumentEmbedder(fake_vectors).embed_corpus([])
        assert matrix.shape == (0, 4)

    def test_rows_match_individual_embed(self, fake_vectors: object) -> None:
        embedder = DocumentEmbedder(fake_vectors)
        documents = [["lung", "failure"], ["vaccine"], ["kidney", "renal"]]
        matrix = embedder.embed_corpus(documents)
        for row, tokens in zip(matrix, documents, strict=True):
            assert np.allclose(row, DocumentEmbedder(fake_vectors).embed(tokens))

    @pytest.mark.parametrize("chunk_size", [1, 2, 3, 100])
    def test_chunking_does_not_change_the_result(
        self, fake_vectors: object, chunk_size: int
    ) -> None:
        documents = [["lung"], ["renal"], ["vaccine"], ["antibody"], ["kidney"]]
        embedder = DocumentEmbedder(fake_vectors)
        chunked = embedder.embed_corpus(documents, chunk_size=chunk_size)
        reference = DocumentEmbedder(fake_vectors).embed_corpus(documents, chunk_size=1000)
        assert np.allclose(chunked, reference)

    def test_oov_counter_resets_per_call(self, fake_vectors: object) -> None:
        embedder = DocumentEmbedder(fake_vectors)
        embedder.embed_corpus([["zzzz"], ["qqqq"]])
        embedder.embed_corpus([["lung"]])
        assert embedder.oov_documents == 0

    def test_oov_ratio_is_warned(
        self, fake_vectors: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            DocumentEmbedder(fake_vectors).embed_corpus([["zzzz"], ["lung"]])
        assert "no in-vocabulary tokens" in caplog.text

    def test_accepts_a_generator(self, fake_vectors: object) -> None:
        matrix = DocumentEmbedder(fake_vectors).embed_corpus(iter([["lung"], ["renal"]]))
        assert matrix.shape == (2, 4)


class TestL2Normalize:
    def test_rows_become_unit_length(self) -> None:
        matrix = np.array([[3, 4, 0, 0], [1, 1, 1, 1]], dtype=np.float32)
        assert np.allclose(np.linalg.norm(l2_normalize(matrix), axis=1), 1.0)

    def test_zero_rows_stay_zero(self) -> None:
        normalised = l2_normalize(np.array([[0, 0, 0, 0]], dtype=np.float32))
        assert np.allclose(normalised, 0.0)

    def test_never_produces_nan(self) -> None:
        matrix = np.array([[3, 4, 0, 0], [0, 0, 0, 0]], dtype=np.float32)
        assert not np.isnan(l2_normalize(matrix)).any()

    def test_direction_is_preserved(self) -> None:
        matrix = np.array([[3, 4, 0, 0]], dtype=np.float32)
        assert np.allclose(l2_normalize(matrix)[0], [0.6, 0.8, 0.0, 0.0], atol=1e-6)

    def test_output_is_float32(self) -> None:
        assert l2_normalize(np.eye(3, dtype=np.float64)).dtype == np.float32

    def test_is_idempotent(self) -> None:
        matrix = np.array([[3, 4, 0, 0], [1, 2, 3, 4]], dtype=np.float32)
        once = l2_normalize(matrix)
        assert np.allclose(l2_normalize(once), once)

    def test_dot_product_of_normalised_rows_is_cosine(self) -> None:
        # This equivalence is what makes ADR-003's single matmul correct.
        a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        b = np.array([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
        cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        normalised = l2_normalize(np.vstack([a, b]))
        assert float(normalised[0] @ normalised[1]) == pytest.approx(cosine, abs=1e-6)
