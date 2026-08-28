"""Structured logging.

Configured once per process. Every pipeline stage is wrapped in
:func:`stage`, which logs entry, exit, wall time, and the resident-memory
delta -- the numbers Architecture.md section 9 is written in and Sprint 8
replaces with measured values.
"""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

from medsearch.runtime import current_rss_mb

_CONFIGURED = False

_PLAIN_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_TIME_FORMAT = "%H:%M:%S"


def configure_logging(level: str = "INFO", *, json_output: bool = False) -> None:
    """Install the root handler. Idempotent -- safe to call from any entrypoint.

    Args:
        level: One of ``DEBUG`` .. ``CRITICAL``.
        json_output: Emit one JSON object per line, for log shipping in Azure.
    """
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger().setLevel(level.upper())
        return

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter() if json_output else _plain_formatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())

    # gensim logs one line per epoch per worker at INFO; that is noise here.
    logging.getLogger("gensim").setLevel(logging.WARNING)
    logging.getLogger("smart_open").setLevel(logging.WARNING)

    _CONFIGURED = True


def _plain_formatter() -> logging.Formatter:
    return logging.Formatter(fmt=_PLAIN_FORMAT, datefmt=_TIME_FORMAT)


class _JsonFormatter(logging.Formatter):
    """Minimal JSON lines formatter -- no dependency on a logging library."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger(name: str) -> logging.Logger:
    """Module-level logger. Use ``get_logger(__name__)``."""
    return logging.getLogger(name)


@contextmanager
def stage(name: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Time a pipeline stage and record its memory cost.

    Emits on exit::

        stage 'train_fasttext' finished in 412.7s | rss 189 -> 1104 MB (+915)

    The RSS delta is the number that matters on an 8 GB machine: it tells you
    which stage is responsible when a run starts swapping.

    Example:
        >>> with stage("build_index"):
        ...     index = build()
    """
    log = logger or get_logger("medsearch.stage")
    rss_before = current_rss_mb()
    started = time.perf_counter()
    log.info("stage %r started | rss %.0f MB", name, rss_before)
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - started
        log.error("stage %r FAILED after %.1fs", name, elapsed, exc_info=True)
        raise
    else:
        elapsed = time.perf_counter() - started
        rss_after = current_rss_mb()
        log.info(
            "stage %r finished in %.1fs | rss %.0f -> %.0f MB (%+.0f)",
            name,
            elapsed,
            rss_before,
            rss_after,
            rss_after - rss_before,
        )
