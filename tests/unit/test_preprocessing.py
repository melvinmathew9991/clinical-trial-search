"""TextPreprocessor and the re-iterable token cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from medsearch.exceptions import ConfigurationError
from medsearch.preprocessing.pipeline import (
    TextPreprocessor,
    TokenCache,
    preprocess_corpus,
)


@pytest.fixture(scope="module")
def preprocessor() -> TextPreprocessor:
    """One instance for the module -- NLTK loading is the expensive part."""
    return TextPreprocessor()


class TestTransform:
    def test_returns_tokens(self, preprocessor: TextPreprocessor) -> None:
        tokens = preprocessor.transform("The patients had severe lung failure in 2020.")
        assert "lung" in tokens
        assert "failure" in tokens

    def test_removes_stopwords(self, preprocessor: TextPreprocessor) -> None:
        tokens = preprocessor.transform("the and of a in for")
        assert tokens == []

    def test_lemmatizes(self, preprocessor: TextPreprocessor) -> None:
        assert "patient" in preprocessor.transform("patients")

    def test_drops_short_tokens(self, preprocessor: TextPreprocessor) -> None:
        assert all(len(t) >= 2 for t in preprocessor.transform("a b cc ddd"))

    def test_empty_input_gives_empty_output(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.transform("") == []

    def test_digits_only_gives_empty_output(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.transform("12345 678") == []

    def test_is_pure(self, preprocessor: TextPreprocessor) -> None:
        text = "Severe respiratory failure"
        first = preprocessor.transform(text)
        second = preprocessor.transform(text)
        assert first == second
        assert text == "Severe respiratory failure"

    def test_document_and_query_share_one_path(self, preprocessor: TextPreprocessor) -> None:
        # PRD F-09: a query and a document must land in the same vector space.
        # The legacy code had two near-duplicate transforms.
        text = "Acute respiratory distress syndrome"
        assert preprocessor.transform(text) == preprocessor.transform(text)

    def test_keep_words_overrides_stopword_removal(self) -> None:
        default = TextPreprocessor()
        keeping = TextPreprocessor(keep_words=["not", "no"])
        assert "not" not in default.transform("not improved")
        assert "not" in keeping.transform("not improved")

    def test_invalid_min_token_length_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="min_token_length"):
            TextPreprocessor(min_token_length=0)


class TestTransformMany:
    def test_is_lazy(self, preprocessor: TextPreprocessor) -> None:
        import types

        result = preprocessor.transform_many(["one text", "another text"])
        assert isinstance(result, types.GeneratorType)

    def test_yields_one_list_per_input(self, preprocessor: TextPreprocessor) -> None:
        assert len(list(preprocessor.transform_many(["a text", "b text", "c text"]))) == 3

    def test_matches_individual_transform(self, preprocessor: TextPreprocessor) -> None:
        texts = ["severe lung failure", "vaccine antibody response"]
        assert list(preprocessor.transform_many(texts)) == [
            preprocessor.transform(t) for t in texts
        ]


class TestTokenCache:
    def test_absent_before_writing(self, tmp_path: Path) -> None:
        assert not TokenCache(tmp_path, "fp", "abstract").exists()

    def test_write_then_exists(self, tmp_path: Path) -> None:
        cache = TokenCache(tmp_path, "fp", "abstract")
        cache.write([["a", "b"], ["c"]])
        assert cache.exists()

    def test_write_returns_count(self, tmp_path: Path) -> None:
        assert TokenCache(tmp_path, "fp", "abstract").write([["a"], ["b"], ["c"]]) == 3

    def test_round_trip(self, tmp_path: Path) -> None:
        cache = TokenCache(tmp_path, "fp", "abstract")
        documents = [["lung", "failure"], ["vaccine"], []]
        cache.write(documents)
        assert list(cache) == documents

    def test_is_re_iterable(self, tmp_path: Path) -> None:
        # gensim iterates the corpus once per epoch; a generator would be
        # exhausted after the first pass (ADR-005).
        cache = TokenCache(tmp_path, "fp", "abstract")
        cache.write([["a"], ["b"]])
        assert list(cache) == list(cache)

    def test_len_counts_documents(self, tmp_path: Path) -> None:
        cache = TokenCache(tmp_path, "fp", "abstract")
        cache.write([["a"], ["b"], ["c"]])
        assert len(cache) == 3

    def test_filename_carries_fingerprint_and_field(self, tmp_path: Path) -> None:
        cache = TokenCache(tmp_path, "abc123", "title")
        assert "abc123" in cache.path.name
        assert "title" in cache.path.name

    def test_different_fingerprints_do_not_collide(self, tmp_path: Path) -> None:
        first = TokenCache(tmp_path, "fp1", "abstract")
        second = TokenCache(tmp_path, "fp2", "abstract")
        assert first.path != second.path

    def test_different_fields_do_not_collide(self, tmp_path: Path) -> None:
        assert (
            TokenCache(tmp_path, "fp", "abstract").path != TokenCache(tmp_path, "fp", "title").path
        )

    def test_unicode_survives(self, tmp_path: Path) -> None:
        cache = TokenCache(tmp_path, "fp", "abstract")
        cache.write([["café", "naïve"]])
        assert list(cache) == [["café", "naïve"]]

    def test_no_temp_file_left_behind(self, tmp_path: Path) -> None:
        cache = TokenCache(tmp_path, "fp", "abstract")
        cache.write([["a"]])
        assert not list(tmp_path.glob("*.tmp"))

    def test_creates_its_directory(self, tmp_path: Path) -> None:
        cache = TokenCache(tmp_path / "nested" / "deeper", "fp", "abstract")
        cache.write([["a"]])
        assert cache.exists()


class TestPreprocessCorpus:
    def test_populates_the_cache(self, tmp_path: Path, preprocessor: TextPreprocessor) -> None:
        cache = TokenCache(tmp_path, "fp", "abstract")
        preprocess_corpus(["severe lung failure", "vaccine response"], cache, preprocessor)
        assert len(list(cache)) == 2

    def test_reuses_an_existing_cache(
        self, tmp_path: Path, preprocessor: TextPreprocessor, caplog: pytest.LogCaptureFixture
    ) -> None:
        cache = TokenCache(tmp_path, "fp", "abstract")
        cache.write([["preexisting"]])
        with caplog.at_level("INFO"):
            preprocess_corpus(["completely different text"], cache, preprocessor)
        assert "Reusing token cache" in caplog.text
        assert list(cache) == [["preexisting"]]

    def test_force_rebuilds(self, tmp_path: Path, preprocessor: TextPreprocessor) -> None:
        cache = TokenCache(tmp_path, "fp", "abstract")
        cache.write([["stale"]])
        preprocess_corpus(["severe lung failure"], cache, preprocessor, force=True)
        assert list(cache) != [["stale"]]

    def test_builds_a_preprocessor_when_none_given(self, tmp_path: Path) -> None:
        cache = TokenCache(tmp_path, "fp", "abstract")
        preprocess_corpus(["severe lung failure"], cache)
        assert len(list(cache)) == 1
