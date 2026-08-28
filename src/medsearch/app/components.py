"""Reusable Streamlit view components.

Split out of ``streamlit_app.py`` so the entrypoint reads as a flow --
controls, query, results -- and each piece of rendering can change without
touching the others.

Nothing here holds state or performs I/O; the entrypoint owns both. That is
what keeps the presentation layer trivially replaceable if the UI ever moves
off Streamlit.
"""

from __future__ import annotations

from typing import Any, Protocol

import streamlit as st

from medsearch.config import get_settings

MODEL_HELP = (
    "Skip-gram learns one vector per word. FastText also learns character "
    "n-grams, so it can produce a vector for a word it never saw during "
    "training -- useful in a domain full of morphological variants."
)

FIELD_HELP = "Abstracts carry more signal; titles give tighter, more literal matches."

UNION_HELP = (
    "Return every trial either keyword or semantic search finds. Roughly doubles "
    "the result count and lifts recall from 0.65 to 0.95, because two thirds of "
    "what the embeddings find is invisible to keyword search."
)

QUERY_PLACEHOLDER = "e.g. lung failure, breathing difficulty, vaccine immunogenicity"


class SearchResultLike(Protocol):
    """Structural type for one ranked result."""

    rank: int
    score: float
    trial_id: str
    title: str
    abstract: str
    publication_date: str


def render_sidebar() -> tuple[str, str, int, bool]:
    """Render the control panel.

    Returns:
        ``(model, field, per_method, union)`` as currently selected.
    """
    with st.sidebar:
        st.header("Search settings")
        union = st.toggle("Keyword + semantic union", value=True, help=UNION_HELP)
        models = ["skipgram", "fasttext"]
        model = st.selectbox(
            "Embedding model",
            options=models,
            index=models.index(get_settings().default_model),
            help=MODEL_HELP,
        )
        field = st.selectbox("Search field", options=["abstract", "title"], help=FIELD_HELP)
        top_n = st.slider(
            "Results per method" if union else "Results",
            min_value=1,
            max_value=25,
            value=10,
        )

        st.divider()
        if union:
            st.caption(
                "Union returns everything either method finds -- about 18 trials "
                "instead of 10. Measured recall 0.955 against 0.648 for keyword "
                "search alone; two thirds of the extra hits are trials keyword "
                "search never returns."
            )
        else:
            st.caption(
                "Semantic only. Vectors are trained in-domain on this corpus, so "
                "clinical terms cluster the way clinicians use them. Note that on "
                "this corpus keyword search alone still ranks better."
            )
    return model, field, top_n, union


def render_sampled_warning(document_count: int) -> None:
    """Warn that the loaded index covers only part of the corpus.

    A sampled index must never look like full coverage. Without this banner the
    UI happily searches 2,000 of 10,666 trials and says nothing.
    """
    st.warning(
        f"Development index: only {document_count:,} documents are searchable. "
        f"Rebuild without `--limit` for full coverage."
    )


def render_empty(reason: str | None) -> None:
    """Explain why a search returned nothing."""
    st.warning(reason or "No matching trials found.")


def render_results_table(frame: Any) -> None:
    """Render the ranked results as a native dataframe.

    Uses ``st.dataframe`` rather than a Plotly ``go.Table`` (ADR-009): the
    legacy UI serialised every returned abstract into a Plotly JSON payload on
    each rerun. The abstract column is omitted here and shown in the
    expanders below, which keeps the table scannable.
    """
    display = frame[["rank", "score", "trial_id", "title", "publication_date"]]
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn("#", width="small"),
            "score": st.column_config.ProgressColumn(
                "Similarity", min_value=0.0, max_value=1.0, format="%.3f"
            ),
            "trial_id": st.column_config.TextColumn("Trial ID", width="small"),
            "title": st.column_config.TextColumn("Title", width="large"),
            "publication_date": st.column_config.TextColumn("Published", width="small"),
        },
    )


def render_abstracts(results: list[SearchResultLike]) -> None:
    """Render each result's full abstract in a collapsed expander."""
    st.subheader("Abstracts")
    for result in results:
        with st.expander(f"{result.rank}. {result.title}  ·  {result.score:.3f}"):
            st.caption(f"{result.trial_id}  ·  published {result.publication_date}")
            st.write(result.abstract)


def render_error(message: str) -> None:
    """Render a domain error as a message, never a traceback (Rules.md 4)."""
    st.error(message)
