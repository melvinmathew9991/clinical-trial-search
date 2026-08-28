"""Command-line interface.

``configure_threads()`` runs at module import, before numpy is pulled in by
anything downstream. That ordering is load-bearing: BLAS reads its thread
limits once, at library load (ADR-008).
"""

from __future__ import annotations

from medsearch.runtime import configure_threads

configure_threads()  # must precede every numpy-importing module below

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import typer  # noqa: E402

from medsearch.config import get_settings  # noqa: E402
from medsearch.exceptions import MedSearchError  # noqa: E402
from medsearch.logging_conf import configure_logging, get_logger  # noqa: E402
from medsearch.runtime import (  # noqa: E402
    cpu_count,
    ensure_nltk_data,
    system_report,
)

app = typer.Typer(
    name="medsearch",
    help="Semantic search over COVID-19 clinical trials.",
    no_args_is_help=True,
    add_completion=False,
)
logger = get_logger("medsearch.cli")

ModelOption = typer.Option("all", "--model", "-m", help="skipgram | fasttext | all")
FieldOption = typer.Option("abstract", "--field", "-f", help="abstract | title")
LimitOption = typer.Option(
    None,
    "--limit",
    "-n",
    help="Row cap for a fast, low-memory development run. Omit for the full corpus.",
)


def _bootstrap() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)


def _fail(exc: MedSearchError) -> None:
    """Render a domain error without a traceback, then exit non-zero."""
    typer.secho(f"\n{exc}\n", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


# ---------------------------------------------------------------- doctor
@app.command()
def doctor(
    full: bool = typer.Option(False, "--full", help="Also check artefact integrity."),
) -> None:
    """Preflight check: cores, memory, disk, NLTK data, artefact budget."""
    _bootstrap()
    settings = get_settings()

    typer.echo(system_report(settings.effective_workers, settings.data_dir).render())

    problems: list[str] = []
    warnings: list[str] = []

    from medsearch.runtime import available_memory_gb

    available = available_memory_gb()
    if available < settings.min_free_memory_gb:
        problems.append(
            f"Only {available:.2f} GB RAM available; training needs "
            f"{settings.min_free_memory_gb:.1f} GB. Close applications or use --limit 2000."
        )
    elif available < settings.warn_free_memory_gb:
        warnings.append(
            f"{available:.2f} GB RAM available. Training will work but may be slow."
        )

    if settings.effective_workers >= cpu_count():
        warnings.append(
            f"workers={settings.effective_workers} on {cpu_count()} logical cores "
            f"leaves nothing for the OS. Set MEDSEARCH_WORKERS=0 for auto."
        )

    predicted_mb = (settings.fasttext_bucket * settings.vector_size * 4) / 1024**2
    typer.echo(f"\n  FastText n-gram matrix: {predicted_mb:.0f} MB (bucket={settings.fasttext_bucket:,})")
    if predicted_mb > settings.max_artefact_mb:
        warnings.append(
            f"Predicted FastText artefact ({predicted_mb:.0f} MB) exceeds the "
            f"{settings.max_artefact_mb} MB budget. Lower MEDSEARCH_FASTTEXT_BUCKET."
        )

    corpus = settings.paths.corpus_file
    typer.echo(f"  Corpus: {'found' if corpus.exists() else 'MISSING'} at {corpus}")
    if not corpus.exists():
        problems.append(f"Corpus missing. Run `make data` or place it at {corpus}.")

    try:
        ensure_nltk_data()
        typer.echo("  NLTK data: ready")
    except Exception as exc:  # pragma: no cover - network dependent
        warnings.append(f"NLTK data unavailable: {exc}")

    if full:
        from medsearch.pipelines.train import artefact_report

        typer.echo("\n  Artefacts:")
        for name, size_mb in artefact_report(settings):
            flag = "  <-- over budget" if size_mb > settings.max_artefact_mb else ""
            typer.echo(f"    {size_mb:8.1f} MB  {name}{flag}")

    for warning in warnings:
        typer.secho(f"\n  WARN  {warning}", fg=typer.colors.YELLOW)
    for problem in problems:
        typer.secho(f"\n  FAIL  {problem}", fg=typer.colors.RED)

    if problems:
        raise typer.Exit(code=1)
    typer.secho("\nPreflight passed.", fg=typer.colors.GREEN)


# ---------------------------------------------------------------- preprocess
@app.command()
def preprocess(
    field: str = FieldOption,
    limit: int | None = LimitOption,
    force: bool = typer.Option(False, "--force", help="Rebuild even if cached."),
) -> None:
    """Clean and tokenize the corpus, caching the result."""
    _bootstrap()
    from medsearch.pipelines.train import run_preprocessing

    try:
        cache, count, _ = run_preprocessing(
            get_settings(), field, limit=limit, force=force  # type: ignore[arg-type]
        )
    except MedSearchError as exc:
        _fail(exc)
    else:
        typer.secho(f"Preprocessed {count} documents -> {cache.path}", fg=typer.colors.GREEN)


# ---------------------------------------------------------------- train
@app.command()
def train(
    model: str = ModelOption,
    field: str = FieldOption,
    limit: int | None = LimitOption,
    force: bool = typer.Option(False, "--force", help="Rebuild the token cache."),
) -> None:
    """Train embedding models on the corpus."""
    _bootstrap()
    from medsearch.pipelines.train import resolve_models, train_one

    settings = get_settings()
    try:
        for name in resolve_models(model):
            outcome = train_one(
                settings, name, field, limit=limit, force=force  # type: ignore[arg-type]
            )
            flag = "  [SAMPLED - development only]" if outcome.sampled else ""
            typer.secho(
                f"  {outcome.model:<9} {outcome.documents:>6} docs | "
                f"vocab {outcome.vocabulary:>6} | {outcome.artefact_mb:>6.1f} MB | "
                f"{outcome.seconds:>6.1f}s{flag}",
                fg=typer.colors.GREEN,
            )
    except MedSearchError as exc:
        _fail(exc)


# ---------------------------------------------------------------- index
index_app = typer.Typer(help="Build and inspect the document index.")
app.add_typer(index_app, name="index")


@index_app.command("build")
def index_build(
    model: str = ModelOption,
    field: str = FieldOption,
    limit: int | None = LimitOption,
) -> None:
    """Embed every document and write a searchable index."""
    _bootstrap()
    from medsearch.pipelines.train import build_index, resolve_models

    settings = get_settings()
    try:
        for name in resolve_models(model):
            idx = build_index(settings, name, field, limit=limit)  # type: ignore[arg-type]
            typer.secho(
                f"  {name:<9} {idx.size:>6} docs x {idx.dim} dims | "
                f"{idx.nbytes / 1024**2:.1f} MB",
                fg=typer.colors.GREEN,
            )
    except MedSearchError as exc:
        _fail(exc)


@index_app.command("info")
def index_info(model: str = ModelOption, field: str = FieldOption) -> None:
    """Show an index's manifest."""
    _bootstrap()
    settings = get_settings()
    directory = settings.paths.index_path(model, field)  # type: ignore[arg-type]
    manifest = directory / "manifest.json"
    if not manifest.exists():
        typer.secho(f"No index at {directory}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(manifest.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- search
@app.command()
def search(
    query: str = typer.Argument(..., help="Free-text search query."),
    model: str = typer.Option("skipgram", "--model", "-m"),
    field: str = FieldOption,
    top_n: int = typer.Option(10, "--top", "-k"),
    limit: int | None = LimitOption,
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Search the corpus."""
    _bootstrap()
    from medsearch.pipelines.train import load_search_engine

    try:
        engine = load_search_engine(
            get_settings(), model, field, limit=limit  # type: ignore[arg-type]
        )
        response = engine.search(query, top_n=top_n)  # type: ignore[attr-defined]
    except MedSearchError as exc:
        _fail(exc)
        return

    if response.is_empty:
        typer.secho(f"No results. {response.reason or ''}", fg=typer.colors.YELLOW)
        return

    if as_json:
        typer.echo(
            json.dumps(
                [
                    {
                        "rank": r.rank,
                        "score": round(r.score, 4),
                        "trial_id": r.trial_id,
                        "title": r.title,
                        "publication_date": r.publication_date,
                    }
                    for r in response.results
                ],
                indent=2,
            )
        )
        return

    typer.echo(f"\nTop {len(response.results)} for {query!r} ({model}/{field})\n")
    for r in response.results:
        typer.secho(f"  {r.rank:>2}. [{r.score:.4f}] {r.trial_id}", fg=typer.colors.CYAN)
        typer.echo(f"      {r.title}")
        typer.echo(f"      {r.truncated_abstract(200)}\n")


# ---------------------------------------------------------------- evaluate
@app.command()
def evaluate(
    model: str = ModelOption,
    field: str = FieldOption,
    eval_file: Path = typer.Option(
        Path("tests/fixtures/eval_queries.json"), "--eval-file"
    ),
) -> None:
    """Compute Recall@k, MRR, and latency. Implemented in Sprint 8."""
    _bootstrap()
    typer.secho(
        "`evaluate` is scheduled for Sprint 8 (see Phases.md). "
        "The CLI surface is reserved; the metrics are not implemented yet.",
        fg=typer.colors.YELLOW,
    )
    raise typer.Exit(code=2)


if __name__ == "__main__":  # pragma: no cover
    app()
