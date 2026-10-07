"""Local cache locations for downloaded and loader-produced data (HANDOFF 4.2, ROLLER step 4).

The cache holds *full* datasets, labels included. In ``local_dir`` mode it gives no isolation
from an agent running as the same user; A2 moves ``split`` and every loader cache into the
warden (risk register, ROLLER). Directories are created owner-only as a basic courtesy.
"""

from __future__ import annotations

import os
from pathlib import Path

CACHE_ENV = "AMX_CACHE"


def cache_dir(*, create: bool = False) -> Path:
    """Root of the amx cache: ``$AMX_CACHE`` if set, else ``~/.cache/amx``."""
    env = os.environ.get(CACHE_ENV)
    root = Path(env).expanduser() if env else Path.home() / ".cache" / "amx"
    if create:
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return root


def datasets_dir(*, create: bool = False) -> Path:
    """Directory where dataset loaders keep fetched files (``<cache>/datasets``)."""
    path = cache_dir(create=create) / "datasets"
    if create:
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path
