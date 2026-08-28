"""Streamlit UI.

Replaces the legacy ``Medical.py``, which reloaded both models and both
21 MB vector CSVs on *every* interaction because Streamlit re-executes the
whole script per rerun and nothing was cached. On a 4-core / 8 GB laptop that
meant several seconds of disk and CPU per keystroke.

Here, models and indexes load once per session via ``st.cache_resource``
(ADR-009, PRD F-32).
"""

from __future__ import annotations

from medsearch.runtime import configure_threads

configure_threads()  # before numpy arrives via any import below

import streamlit as st  # noqa: E402

from medsearch.config import get_settings  # noqa: E402
from medsearch.exceptions import MedSearchError  # noqa: E402
from medsearch.logging_conf import configure_logging  # noqa: E402

st.set_page_config(
    page_title="Clinical Trial Search",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading model and index...")
def _load_engine(model: str, field: str) -> object:
    """Build a search engine once per (model, field) per session.

    ``cache_resource`` is the correct decorator here rather than
    ``cache_data``: the engine holds a memory-mapped index and a gensim
    ``KeyedVectors``, neither of which should be pickled or copied.
    """
    from medsearch.pipelines.train import load_search_engine

    settings = get_settings()
    configure_logging(settings.log_level)
    return load_search_engine(settings, model, field)  # type: ignore[arg-type]


def _sidebar() -> tuple[str, str, int]:
    """Render controls and return the current selection."""
    with st.sidebar:
        st.header("Search settings")
        model = st.selectbox(
            "Embedding model",
            options=["skipgram", "fasttext"],
            help=(
                "Skip-gram learns one vector per word. FastText also learns "
                "character n-grams, so it can produce a vector for a word it "
                "never saw during training."
            ),
        )
        field = st.selectbox(
            "Search field",
            options=["abstract", "title"],
            help="Abstracts carry more signal; titles give tighter, more literal matches.",
        )
        top_n = st.slider("Results", min_value=1, max_value=25, value=10)

        st.divider()
        st.caption(
            "Vectors are trained in-domain on this corpus, so clinical terms "
            "cluster the way clinicians use them rather than the way news text does."
        )
    return model, field, top_n


def _render_results(response: object) -> None:
    """Render a search response as a table plus expandable abstracts."""
    if response.is_empty:  # type: ignore[attr-defined]
        st.warning(response.reason or "No matching trials found.")  # type: ignore[attr-defined]
        return

    results = response.results  # type: ignore[attr-defined]
    st.success(f"{len(results)} trials, ranked by semantic similarity.")

    # st.dataframe instead of a Plotly go.Table: the legacy UI serialised
    # every returned abstract into a Plotly JSON payload on each rerun.
    frame = response.to_frame()  # type: ignore[attr-defined]
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

    st.subheader("Abstracts")
    for result in results:
        with st.expander(f"{result.rank}. {result.title}  ·  {result.score:.3f}"):
            st.caption(f"{result.trial_id}  ·  published {result.publication_date}")
            st.write(result.abstract)


def main() -> None:
    """Application entrypoint."""
    st.title("Clinical Trial Search")
    st.caption(
        "Semantic search over COVID-19 clinical trials using in-domain "
        "Word2Vec and FastText embeddings."
    )

    model, field, top_n = _sidebar()

    query = st.text_input(
        "Search",
        placeholder="e.g. lung failure, breathing difficulty, vaccine immunogenicity",
        help="Free text. Results are ranked by meaning, not keyword overlap.",
    )

    if not query:
        st.info("Enter a query above to search the corpus.")
        return

    try:
        engine = _load_engine(model, field)
        response = engine.search(query, top_n=top_n)  # type: ignore[attr-defined]
    except MedSearchError as exc:
        # Domain errors carry an actionable message; never show a traceback
        # to the browser (Rules.md section 4).
        st.error(str(exc))
        return

    _render_results(response)


if __name__ == "__main__":
    main()
