"""Integration fixtures.

These tests train real models, so they touch the memory preflight. That
preflight reads *ambient* free RAM, which makes any test behind it a function
of whatever else the machine is doing -- and it duly failed CI intermittently,
looking like test-order dependence when it was memory pressure (see
``TestFastTextMemoryGuardScales`` in ``tests/unit/test_regressions.py``).

The guard's own behaviour is pinned by unit tests that stub the reading
directly. Here it is held at a fixed generous value so these tests measure the
pipeline and nothing else.
"""

from __future__ import annotations

import pytest

#: Comfortably above the full-corpus requirement, so the preflight never fires.
_STUBBED_FREE_GB = 8.0


@pytest.fixture(autouse=True)
def _stable_memory_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hold free-RAM readings steady for every integration test."""
    from medsearch import runtime
    from medsearch.embeddings import trainer

    monkeypatch.setattr(runtime, "available_memory_gb", lambda: _STUBBED_FREE_GB)
    monkeypatch.setattr(trainer, "available_memory_gb", lambda: _STUBBED_FREE_GB)
