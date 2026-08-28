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
from typing import cast  # noqa: E402

import typer  # noqa: E402

from medsearch.config import FieldName, ModelName, Pooling, Settings, get_settings  # noqa: E402
from medsearch.exceptions import (  # noqa: E402
    CORPUS_SOURCE_URL,
    ConfigurationError,
    MedSearchError,
)
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


VALID_MODELS = ("skipgram", "fasttext")
VALID_FIELDS = ("abstract", "title")
VALID_POOLING = ("mean", "sif")

PoolingOption = typer.Option(
    None,
    "--pooling",
    "-p",
    help="mean | sif. Omit to use MEDSEARCH_POOLING (default mean).",
)


def _as_model(value: str, *, allow_all: bool = False) -> str:
    """Validate a --model value at the CLI boundary.

    Without this, an unrecognised value travelled all the way into the
    pipeline before failing on a path that did not exist. Validating here
    also narrows `str` to the Literal the pipeline signatures declare.
    """
    allowed = (*VALID_MODELS, "all") if allow_all else VALID_MODELS
    if value not in allowed:
        raise ConfigurationError(f"Unknown model {value!r}.\n  Valid choices: {', '.join(allowed)}")
    return value


def _as_field(value: str) -> FieldName:
    """Validate a --field value at the CLI boundary."""
    if value not in VALID_FIELDS:
        raise ConfigurationError(
            f"Unknown field {value!r}.\n  Valid choices: {', '.join(VALID_FIELDS)}"
        )
    return cast(FieldName, value)


def _as_pooling(value: str | None) -> Pooling | None:
    """Validate a --pooling value; None means fall back to settings."""
    if value is None:
        return None
    if value not in VALID_POOLING:
        raise ConfigurationError(
            f"Unknown pooling {value!r}.\n  Valid choices: {', '.join(VALID_POOLING)}"
        )
    return cast(Pooling, value)


def _bootstrap() -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)


def _fail(exc: MedSearchError) -> None:
    """Render a domain error without a traceback, then exit non-zero."""
    typer.secho(f"\n{exc}\n", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


# ---------------------------------------------------------------- doctor
def _check_resources(settings: Settings) -> tuple[list[str], list[str]]:
    """Check RAM and worker count against what training needs.

    Returns:
        ``(problems, warnings)``. A problem fails the preflight; a warning is
        reported and tolerated.
    """
    from medsearch.runtime import available_memory_gb

    problems: list[str] = []
    warnings: list[str] = []

    available = available_memory_gb()
    if available < settings.min_free_memory_gb:
        problems.append(
            f"Only {available:.2f} GB RAM available; training needs "
            f"{settings.min_free_memory_gb:.1f} GB. Close applications or use --limit 2000."
        )
    elif available < settings.warn_free_memory_gb:
        warnings.append(f"{available:.2f} GB RAM available. Training will work but may be slow.")

    if settings.effective_workers >= cpu_count():
        warnings.append(
            f"workers={settings.effective_workers} on {cpu_count()} logical cores "
            f"leaves nothing for the OS. Set MEDSEARCH_WORKERS=0 for auto."
        )
    return problems, warnings


def _report_artefacts(settings: Settings) -> list[str]:
    """Print artefact sizes and integrity findings for ``doctor --full``.

    Returns:
        Problems worth failing the preflight over -- integrity findings at
        ERROR severity. Warnings print here and are not returned, since
        nothing downstream acts on them.
    """
    from medsearch.pipelines.integrity import Severity, check_artefacts
    from medsearch.pipelines.train import artefact_report

    typer.echo("\n  Artefacts:")
    for name, size_mb in artefact_report(settings):
        flag = "  <-- over budget" if size_mb > settings.max_artefact_mb else ""
        typer.echo(f"    {size_mb:8.1f} MB  {name}{flag}")

    typer.echo("\n  Integrity:")
    try:
        findings = check_artefacts(settings)
    except MedSearchError as exc:
        typer.secho(f"    could not verify: {exc}", fg=typer.colors.YELLOW)
        return []

    if not findings:
        typer.secho(
            "    every artefact is consistent with its model and the live corpus",
            fg=typer.colors.GREEN,
        )
    problems: list[str] = []
    for finding in findings:
        colour = typer.colors.RED if finding.severity is Severity.ERROR else typer.colors.YELLOW
        typer.secho(f"  {finding.render()}", fg=colour)
        if finding.severity is Severity.ERROR:
            problems.append(f"{finding.code}: {finding.message}")
    return problems


@app.command()
def doctor(
    full: bool = typer.Option(False, "--full", help="Also check artefact integrity."),
) -> None:
    """Preflight check: cores, memory, disk, NLTK data, artefact budget."""
    _bootstrap()
    settings = get_settings()

    typer.echo(system_report(settings.effective_workers, settings.data_dir).render())

    problems, warnings = _check_resources(settings)

    predicted_mb = (settings.fasttext_bucket * settings.vector_size * 4) / 1024**2
    typer.echo(
        f"\n  FastText n-gram matrix: {predicted_mb:.0f} MB (bucket={settings.fasttext_bucket:,})"
    )
    if predicted_mb > settings.max_artefact_mb:
        warnings.append(
            f"Predicted FastText artefact ({predicted_mb:.0f} MB) exceeds the "
            f"{settings.max_artefact_mb} MB budget. Lower MEDSEARCH_FASTTEXT_BUCKET."
        )

    corpus = settings.paths.corpus_file
    typer.echo(f"  Corpus: {'found' if corpus.exists() else 'MISSING'} at {corpus}")
    if not corpus.exists():
        problems.append(
            f"Corpus missing at {corpus}.\n"
            f"        Download the Dimensions COVID-19 export, save the CSV there:\n"
            f"        {CORPUS_SOURCE_URL}\n"
            f"        (`make data` migrates a local legacy Part_1 tree and "
            f"cannot help a fresh clone.)"
        )

    try:
        ensure_nltk_data()
        typer.echo("  NLTK data: ready")
    except Exception as exc:  # pragma: no cover - network dependent
        warnings.append(f"NLTK data unavailable: {exc}")

    if full:
        problems.extend(_report_artefacts(settings))

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
            get_settings(), _as_field(field), limit=limit, force=force
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
        for name in resolve_models(_as_model(model, allow_all=True)):
            outcome = train_one(settings, name, _as_field(field), limit=limit, force=force)
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
    pooling: str | None = PoolingOption,
) -> None:
    """Embed every document and write a searchable index."""
    _bootstrap()
    from medsearch.pipelines.train import build_index, resolve_models

    settings = get_settings()
    try:
        mode = _as_pooling(pooling)
        for name in resolve_models(_as_model(model, allow_all=True)):
            idx = build_index(settings, name, _as_field(field), limit=limit, pooling=mode)
            typer.secho(
                f"  {name:<9} {idx.size:>6} docs x {idx.dim} dims | "
                f"{idx.nbytes / 1024**2:.1f} MB | pooling={idx.pooling}",
                fg=typer.colors.GREEN,
            )
    except MedSearchError as exc:
        _fail(exc)


@index_app.command("info")
def index_info(model: str = ModelOption, field: str = FieldOption) -> None:
    """Show an index's manifest."""
    _bootstrap()
    settings = get_settings()
    directory = settings.paths.index_path(cast(ModelName, _as_model(model)), _as_field(field))
    manifest = directory / "manifest.json"
    if not manifest.exists():
        typer.secho(f"No index at {directory}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    typer.echo(manifest.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- search
@app.command()
def search(
    query: str = typer.Argument(..., help="Free-text search query."),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="skipgram | fasttext. Defaults to settings.default_model (fasttext).",
    ),
    field: str = FieldOption,
    top_n: int = typer.Option(10, "--top", "-k"),
    limit: int | None = LimitOption,
    pooling: str | None = PoolingOption,
    union: bool | None = typer.Option(
        None,
        "--union/--no-union",
        help="Return the union of keyword and semantic results (~18 docs, recall 0.955).",
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit JSON instead of a table."),
) -> None:
    """Search the corpus."""
    _bootstrap()
    from medsearch.pipelines.train import load_search_engine

    try:
        settings = get_settings()
        # Resolve from settings rather than a literal default in the signature,
        # so changing the shipped model is a config change and not a hunt
        # through every entry point (PRD 8.4 moved it to FastText).
        chosen = settings.default_model if model is None else _as_model(model)
        use_union = settings.union_retrieval if union is None else union
        if use_union:
            from medsearch.pipelines.train import load_union_retriever

            retriever = load_union_retriever(
                settings, cast(ModelName, chosen), _as_field(field), limit=limit
            )
            response = retriever.search(query, per_method=top_n)  # type: ignore[attr-defined]
        else:
            engine = load_search_engine(
                settings,
                cast(ModelName, chosen),
                _as_field(field),
                limit=limit,
                pooling=_as_pooling(pooling),
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

    typer.echo(f"\nTop {len(response.results)} for {query!r} ({chosen}/{field})\n")
    for r in response.results:
        typer.secho(f"  {r.rank:>2}. [{r.score:.4f}] {r.trial_id}", fg=typer.colors.CYAN)
        typer.echo(f"      {r.title}")
        typer.echo(f"      {r.truncated_abstract(200)}\n")


# ---------------------------------------------------------------- evaluate
#: Module-level singleton so the default is not a call evaluated at import
#: time in the signature (ruff B008).
EvalFileOption = typer.Option(Path("tests/fixtures/eval_queries.json"), "--eval-file")


@app.command()
def evaluate(
    model: str = ModelOption,
    field: str = FieldOption,
    eval_file: Path = EvalFileOption,
    pooling: str | None = PoolingOption,
    no_baseline: bool = typer.Option(
        False, "--no-baseline", help="Skip the TF-IDF keyword comparison."
    ),
) -> None:
    """Measure Recall@k, MRR@k and latency against the TF-IDF baseline.

    Requires a human-labelled evaluation set. Relevance judgements are never
    generated automatically -- run `python scripts/make_eval_candidates.py` to
    produce a candidate sheet to label.

    Exits 1 when a PRD target is missed, so this can gate a release.
    """
    _bootstrap()
    from medsearch.pipelines.evaluate import check_targets, run_evaluation

    models = ["skipgram", "fasttext"] if model == "all" else [_as_model(model)]

    try:
        report = run_evaluation(
            get_settings(),
            eval_path=eval_file,
            field=_as_field(field),
            models=cast("list[ModelName]", models),
            include_baseline=not no_baseline,
            pooling=_as_pooling(pooling),
        )
    except MedSearchError as exc:
        _fail(exc)
        return

    typer.echo(f"\nEvaluation over {report['eval_queries']} labelled queries\n")
    for entry in report["results"]:
        line = (
            f"  {entry['method']:<22} "
            f"R@10 {float(entry['recall_at'].get('10', 0)):.3f}  "
            f"MRR@10 {float(entry['mrr_at'].get('10', 0)):.3f}  "
            f"p95 {float(entry['latency_ms'].get('p95', 0)):>7.2f} ms  "
            f"docs {float(entry.get('results_shown', 0)):>4.1f}"
        )
        colour = typer.colors.CYAN if entry["method"] == "tfidf-baseline" else typer.colors.GREEN
        typer.secho(line, fg=colour)

    # The union returns roughly twice as many documents as the single rankers.
    # Printing recall without that number invites reading it as a like-for-like
    # win, which it is not -- it is a different, larger budget.
    typer.echo("")
    typer.secho(
        "  'docs' is the mean trials returned per query. Union methods return the",
        fg=typer.colors.BRIGHT_BLACK,
    )
    typer.secho(
        "  union of two top-10 lists and are scored to depth 20; the single",
        fg=typer.colors.BRIGHT_BLACK,
    )
    typer.secho("  rankers are scored to depth 10.", fg=typer.colors.BRIGHT_BLACK)

    failures = check_targets(report)
    typer.echo("")
    if failures:
        for failure in failures:
            typer.secho(f"  MISS  {failure}", fg=typer.colors.YELLOW)
        typer.secho(
            "\n  A missed target is a finding, not a crash. See PRD "
            "section 8 for what has already been tried and measured.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)

    typer.secho("  All PRD targets met.", fg=typer.colors.GREEN)


if __name__ == "__main__":  # pragma: no cover
    app()
