"""The text preprocessing pipeline.

One class, one transform, used for **both** corpus documents and user queries
(PRD F-09). The legacy code had two near-duplicate paths -- ``preprocessing()``
for documents and ``preprocessing_input()`` for queries -- which is how a
query and a document could drift out of the same vector space.

Streaming is the other concern: :meth:`TextPreprocessor.transform_many` is a
generator, so gensim consumes tokens without the ~600 MB cost of materialising
every token list at once (ADR-005).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Final

from medsearch.exceptions import ConfigurationError
from medsearch.logging_conf import get_logger
from medsearch.preprocessing.normalizer import clean_text
from medsearch.runtime import ensure_nltk_data

logger = get_logger(__name__)


#: Negation and quantity terms that NLTK classes as English stopwords but
#: which invert clinical meaning. Without these, "no evidence of thrombosis"
#: and "evidence of thrombosis" tokenise identically -- the retrieval system
#: cannot tell a ruled-out finding from a confirmed one.
#:
#: The old behaviour was worse than uniformly wrong, it was *inconsistent*:
#: "no" and "not" were dropped as stopwords while "without", "never", "none",
#: "absent" and "negative" survived, so half the negation vocabulary was
#: silently discarded and half was not. PRD F-12 anticipated this and the hook
#: below was left unwired; the domain audit is what finally exercised it.
#:
#: Deliberately narrow. Every word here costs vocabulary and appears in almost
#: every abstract, so this is not the place for general clinical terms.
CLINICAL_KEEP_WORDS: Final[tuple[str, ...]] = (
    "no",
    "not",
    "nor",
    "none",
    "cannot",
    "against",
    "before",
    "after",
    "during",
    "below",
    "above",
    "under",
    "over",
    "off",
    "out",
)


class TextPreprocessor:
    """Normalise, tokenize, remove stopwords, and lemmatize.

    NLTK resources are loaded once in ``__init__`` rather than at module
    import, so constructing the object is the only place that can touch the
    network -- and :func:`~medsearch.runtime.ensure_nltk_data` makes that a
    no-op after the first run.

    Args:
        keep_words: Words to retain even if they are English stopwords.
            Defaults to :data:`CLINICAL_KEEP_WORDS`, the negation allowlist
            required by PRD F-12. Pass an explicit empty tuple to restore the
            plain-English behaviour.
        min_token_length: Tokens shorter than this are dropped. Single
            characters survive normalisation but carry no signal.

    Example:
        >>> pre = TextPreprocessor()
        >>> pre.transform("The patients had severe lung failure in 2020.")
        ['patient', 'severe', 'lung', 'failure']
    """

    def __init__(
        self,
        *,
        keep_words: Sequence[str] | None = None,
        min_token_length: int = 2,
    ) -> None:
        if min_token_length < 1:
            raise ConfigurationError("min_token_length must be >= 1")

        ensure_nltk_data()
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize

        self._tokenize = word_tokenize
        self._lemmatizer = WordNetLemmatizer()
        self._min_token_length = min_token_length

        stop = set(stopwords.words("english"))
        retained = CLINICAL_KEEP_WORDS if keep_words is None else keep_words
        stop -= {w.lower() for w in retained}
        # frozenset: O(1) membership, and immutable so it cannot be mutated
        # by a caller holding a reference to the preprocessor.
        self._stopwords: frozenset[str] = frozenset(stop)

    def transform(self, text: str) -> list[str]:
        """Convert one string to its final token list.

        Pure: takes a string, returns a new list, mutates nothing.

        Args:
            text: Raw document or query text.

        Returns:
            Lemmatized, stopword-filtered tokens. Empty list for empty or
            fully-filtered input -- callers must handle that case rather than
            assume at least one token.
        """
        cleaned = clean_text(text)
        if not cleaned:
            return []
        return [
            self._lemmatizer.lemmatize(token)
            for token in self._tokenize(cleaned)
            if token not in self._stopwords and len(token) >= self._min_token_length
        ]

    def transform_many(self, texts: Iterable[str]) -> Iterator[list[str]]:
        """Lazily transform an iterable of strings.

        A generator by design. gensim re-iterates its corpus once per epoch,
        so pair this with :class:`TokenCache` when training for more than one
        epoch, rather than re-running the transform each pass.
        """
        for text in texts:
            yield self.transform(text)


class TokenCache:
    """JSONL cache of preprocessed tokens, keyed by corpus fingerprint.

    Preprocessing 10,666 abstracts takes about four minutes; a cache hit takes
    about three seconds. The fingerprint in the filename means a changed corpus
    can never silently reuse stale tokens.

    The file is also re-iterable, which is what gensim needs for multi-epoch
    training without holding every token list in RAM (ADR-005).
    """

    def __init__(self, directory: Path, fingerprint: str, field: str) -> None:
        self._path = directory / f"{fingerprint}.{field}.tokens.jsonl"
        self._directory = directory

    @property
    def path(self) -> Path:
        """Location of the cache file."""
        return self._path

    def exists(self) -> bool:
        """True when a cache for this corpus and field is present."""
        return self._path.exists()

    def write(self, documents: Iterable[Sequence[str]]) -> int:
        """Stream token lists to disk, one JSON array per line.

        Returns:
            Number of documents written.
        """
        self._directory.mkdir(parents=True, exist_ok=True)
        count = 0
        # Write to a temp file then rename, so an interrupted run never leaves
        # a truncated cache that a later run would trust.
        temp = self._path.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            for tokens in documents:
                handle.write(json.dumps(list(tokens), ensure_ascii=False))
                handle.write("\n")
                count += 1
        temp.replace(self._path)
        logger.info("Cached %d preprocessed documents to %s", count, self._path.name)
        return count

    def __iter__(self) -> Iterator[list[str]]:
        """Re-iterable stream of cached token lists.

        gensim calls this once per epoch. Memory stays flat regardless of
        corpus size because only one line is decoded at a time.
        """
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)

    def __len__(self) -> int:
        """Number of cached documents. Requires a full scan."""
        with self._path.open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())


def preprocess_corpus(
    texts: Iterable[str],
    cache: TokenCache,
    preprocessor: TextPreprocessor | None = None,
    *,
    force: bool = False,
) -> TokenCache:
    """Preprocess a corpus into a re-iterable token cache.

    Args:
        texts: Raw document texts.
        cache: Destination cache, keyed by corpus fingerprint.
        preprocessor: Reused if supplied; constructed otherwise.
        force: Rebuild even when a valid cache exists.

    Returns:
        The populated cache, ready to hand to a trainer.
    """
    if cache.exists() and not force:
        logger.info("Reusing token cache %s", cache.path.name)
        return cache

    pre = preprocessor or TextPreprocessor()
    cache.write(pre.transform_many(texts))
    return cache
