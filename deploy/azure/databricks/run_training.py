"""Databricks entrypoint for model training.

This file replaces the legacy Part_2 ``main.py``, which began with::

    # MAGIC %run ./training_model

...referencing a ``training_model`` module that **does not exist anywhere in
the repository**. The notebook could never have run as shipped.

The fix is structural rather than a missing file: the training logic now lives
in the installed ``medsearch`` wheel, so Databricks, the CLI, and the tests all
execute the same code. This script is a thin argument-parsing shim.

Deployment:
    1. ``python -m build --wheel``
    2. Upload the wheel to ``dbfs:/FileStore/medsearch/wheels/``
    3. Upload this file to ``dbfs:/FileStore/medsearch/jobs/``
    4. The ADF pipeline ``train-embeddings`` invokes it as a Spark Python task.

Storage access uses the cluster's managed identity via ABFS. No SAS token
appears here or in any tracked file -- the legacy version pasted live tokens
with read/write/delete rights straight into ``read_data.py`` and ``top_n.py``.
"""

from __future__ import annotations

import argparse
import sys

from medsearch.runtime import configure_threads

configure_threads()

from medsearch.config import Settings  # noqa: E402
from medsearch.logging_conf import configure_logging, get_logger  # noqa: E402
from medsearch.pipelines.train import resolve_models, train_one  # noqa: E402

logger = get_logger("medsearch.databricks.train")


def build_settings(mount: str) -> Settings:
    """Point the pipeline at the DBFS mount backed by blob storage.

    The mount is configured once at the workspace level with a managed
    identity; this code never sees a credential.
    """
    return Settings(
        data_dir=f"{mount}/data",  # type: ignore[arg-type]
        model_dir=f"{mount}/models",  # type: ignore[arg-type]
        report_dir=f"{mount}/reports",  # type: ignore[arg-type]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="all", choices=["skipgram", "fasttext", "all"])
    parser.add_argument("--field", default="abstract", choices=["abstract", "title"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mount", default="/dbfs/mnt/medsearch")
    args = parser.parse_args(argv)

    settings = build_settings(args.mount)
    configure_logging(settings.log_level, json_output=True)

    logger.info(
        "Databricks training run: model=%s field=%s limit=%s mount=%s",
        args.model,
        args.field,
        args.limit,
        args.mount,
    )

    failures = 0
    for name in resolve_models(args.model):
        try:
            outcome = train_one(settings, name, args.field, limit=args.limit)
        except Exception:
            logger.exception("Training failed for %s", name)
            failures += 1
        else:
            logger.info(
                "Trained %s: %d docs, vocab %d, %.1f MB, %.1fs",
                outcome.model,
                outcome.documents,
                outcome.vocabulary,
                outcome.artefact_mb,
                outcome.seconds,
            )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
