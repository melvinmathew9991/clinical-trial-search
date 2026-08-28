"""Streamlit entrypoint.

Replaces the legacy ``Medical.py``, which reloaded both models and both 21 MB
vector CSVs on *every* interaction, because Streamlit re-executes the whole
script per rerun and nothing was cached. On a 4-core / 8 GB laptop that meant
several seconds of disk and CPU per keystroke.

Here the models and index load once per session via ``st.cache_resource``
(PRD F-32). Rendering lives in :mod:`medsearch.app.components`; this module is
just the flow.
"""

from __future__ import annotations

from medsearch.runtime import configure_threads

configure_threads()  # before numpy arrives via any import below

import streamlit as st  # noqa: E402

from medsearch.app.components import (  # noqa: E402
    QUERY_PLACEHOLDER,
    render_abstracts,
    render_empty,
    render_error,
    render_results_table,
    render_sampled_warning,
    render_sidebar,
)
from medsearch.config import get_settings  # noqa: E402
from medsearch.exceptions import MedSearchError  # noqa: E402
from medsearch.logging_conf import configure_logging  # noqa: E402

st.set_page_config(
    page_title="Clinical Trial Search",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading keyword and semantic indexes...")
def load_union(model: str, field: str) -> object:
    """Build the union retriever once per session.

    Union retrieval is the default because it lifts Recall@10 from 0.648 to
    0.955 (PRD 8.3). It returns about 18 results instead of 10; for a
    researcher scanning a list of trials that trade is worth making, and MRR is
    statistically unchanged so the top of the list is no worse.
    """
    from medsearch.pipelines.train import load_union_retriever

    settings = get_settings()
    configure_logging(settings.log_level)
    return load_union_retriever(
        settings,
        model,  # type: ignore[arg-type]
        field,  # type: ignore[arg-type]
    )


@st.cache_resource(show_spinner="Loading model and index...")
def load_engine(model: str, field: str) -> object:
    """Build a search engine once per (model, field) per session.

    ``cache_resource`` rather than ``cache_data``: the engine holds a
    memory-mapped index and a gensim ``KeyedVectors``, neither of which should
    be pickled or copied.
    """
    from medsearch.pipelines.train import load_search_engine

    settings = get_settings()
    configure_logging(settings.log_level)
    return load_search_engine(
        settings,
        model,  # type: ignore[arg-type]
        field,  # type: ignore[arg-type]
    )


def main() -> None:
    """Application entrypoint."""
    st.title("Clinical Trial Search")
    st.caption(
        "Semantic search over COVID-19 clinical trials using in-domain "
        "Word2Vec and FastText embeddings."
    )

    model, field, top_n, union = render_sidebar()

    query = st.text_input(
        "Search",
        placeholder=QUERY_PLACEHOLDER,
        help="Free text. Results are ranked by meaning, not keyword overlap.",
    )

    if not query:
        st.info("Enter a query above to search the corpus.")
        return

    try:
        if union:
            retriever = load_union(model, field)
            response = retriever.search(query, per_method=top_n)  # type: ignore[attr-defined]
        else:
            engine = load_engine(model, field)
            if getattr(engine, "is_sampled", False):
                render_sampled_warning(engine.size)  # type: ignore[attr-defined]
            response = engine.search(query, top_n=top_n)  # type: ignore[attr-defined]
    except MedSearchError as exc:
        # Domain errors carry an actionable message; a traceback must never
        # reach the browser (Rules.md section 4).
        render_error(str(exc))
        return

    if response.is_empty:
        render_empty(response.reason)
        return

    mode = "keyword + semantic union" if union else "semantic similarity"
    st.success(f"{len(response.results)} trials, ranked by {mode}.")
    render_results_table(response.to_frame())
    render_abstracts(response.results)


if __name__ == "__main__":
    main()
