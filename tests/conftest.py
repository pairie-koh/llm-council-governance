"""Shared test fixtures."""

import pytest

import backend.governance.base as governance_base


@pytest.fixture(autouse=True)
def disable_stage1_shared_cache(monkeypatch):
    """Disable the cross-structure Stage 1 disk cache in tests.

    Tests mock query_models_parallel with per-test responses; a shared disk
    cache keyed on prompt text would leak one test's mocked responses into
    another and skip API calls that call-sequence tests assert on.
    """
    monkeypatch.setattr(governance_base, "STAGE1_SHARED_CACHE", False)
    governance_base._stage1_locks.clear()
