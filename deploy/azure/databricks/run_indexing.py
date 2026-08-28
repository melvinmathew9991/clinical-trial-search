"""Databricks entrypoint for index building.

Runs after `run_training.py` in the `train-embeddings` ADF pipeline. Split
into a separate task deliberately: training and indexing have different peak
memory profiles, and separate processes let the cluster reclaim training
memory before indexing starts (Architecture.md section 9).
"""

from __future__ import annotations

import argparse
import sys

from medsearch.runtime import configure_threads

configure_threads()

from medsearch.logging_conf import configure_logging, get_logger  # noqa: E402
from medsearch.pipelines.train import build_index, resolve_models  # noqa: E402

from run_training import build_settings  # noqa: E402

logger = get_logger("medsearch.databricks.index")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="all", choices=["skipgram", "fasttext", "all"])
    parser.add_argument("--field", default="abstract", choices=["abstract", "title"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mount", default="/dbfs/mnt/medsearch")
    args = parser.parse_args(argv)

    settings = build_settings(args.mount)
    configure_logging(settings.log_level, json_output=True)

    failures = 0
    for name in resolve_models(args.model):
        try:
            index = build_index(settings, name, args.field, limit=args.limit)
        except Exception:
            logger.exception("Index build failed for %s", name)
            failures += 1
        else:
            logger.info(
                "Indexed %s: %d docs x %d dims (%.1f MB)",
                name,
                index.size,
                index.dim,
                index.nbytes / 1024**2,
            )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
