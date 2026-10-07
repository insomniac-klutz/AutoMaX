from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolated_vault(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every test gets its own throwaway vault root holding synthetic data only."""
    root = tmp_path_factory.mktemp("amxv")
    monkeypatch.setenv("AMX_VAULT", str(root))
    monkeypatch.delenv("AMX_FREEZE_TOKEN", raising=False)
    os.environ.setdefault("PYTHONHASHSEED", "0")
