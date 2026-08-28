"""Medical Embeddings Search.

Semantic retrieval over COVID-19 clinical trials using in-domain Word2Vec
(Skip-gram) and FastText embeddings.

Read the project docs in this order before working on the code:
``PRD.md`` -> ``Architecture.md`` -> ``Rules.md`` -> ``Phases.md`` -> ``Memory.md``.

Example:
    >>> from medsearch.config import get_settings
    >>> from medsearch.pipelines import load_search_engine
    >>> engine = load_search_engine(get_settings(), "skipgram", "abstract")
    >>> response = engine.search("lung failure", top_n=5)
"""

from __future__ import annotations

__version__ = "0.5.0"

__all__ = ["__version__"]
