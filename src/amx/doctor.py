"""``amx doctor``: environment check (HANDOFF 5.3)."""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import sys
from pathlib import Path
from typing import Any

import amx

PACKAGES = ("numpy", "scipy", "pandas", "pyarrow", "scikit-learn", "pydantic", "typer", "joblib")
OPTIONAL = ("lightgbm", "xgboost", "catboost", "autogluon", "sentence-transformers", "torch")


def _version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def diagnose(repo_root: Path | None = None) -> dict[str, Any]:
    """Collect environment facts and problems. ``ok`` is False when a problem blocks use."""
    from amx.split.vault import LocalVault

    problems: list[str] = []
    warnings: list[str] = []
    vault = LocalVault()
    root = vault.root
    vault_info: dict[str, Any] = {"mode": "local_dir", "root": str(root), "exists": root.exists()}
    if root.exists():
        mode = root.stat().st_mode & 0o777
        vault_info["permissions"] = oct(mode)
        if mode & 0o077:
            problems.append(f"vault root {root} is readable by group/others ({oct(mode)})")
    note = vault.isolation_warning()
    if note:
        warnings.append(note)
    if os.environ.get("AMX_FREEZE_TOKEN"):
        problems.append("AMX_FREEZE_TOKEN is set in this environment; it must never reach an agent")
    repo = repo_root or Path.cwd()
    hooks = repo / "claude-plugin" / "hooks" / "hooks.json"
    return {
        "ok": not problems,
        "amx_version": amx.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "euid": os.geteuid() if hasattr(os, "geteuid") else None,
        "packages": {p: _version(p) for p in PACKAGES},
        "extras": {p: _version(p) for p in OPTIONAL},
        "vault": vault_info,
        "hooks": {"plugin_hooks": hooks.exists(), "note": "plugin and hooks arrive in A2"},
        "problems": problems,
        "warnings": warnings,
    }
