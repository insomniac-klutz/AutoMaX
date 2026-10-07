"""The ``local_dir`` vault (HANDOFF 8.4, ROLLER step 5).

The vault holds calib and sealed inputs and labels outside the run tree, plus the per-run
HMAC key for the manifest digests (C13), the TaskSpec snapshot and the warden's append-only
logs. Layout::

    <root>/<run_id>/            0700
        folds/calib.parquet     0600
        folds/sealed.parquet    0600
        hmac.key                0600
        taskspec.yaml, ...      0600

``root`` is ``$AMX_VAULT`` or ``~/.amx/vault``. amx creates a missing root with mode 0700 but
never changes an existing one (it may be shared, like ``/tmp``); a shared root owned by another
user is refused. Run and ``folds`` directories are always forced to 0700.

``local_dir`` is a convenience mode only: it is not safe against a determined agent running
as the same user (see :meth:`LocalVault.isolation_warning`). Container mode arrives in A2.
"""

from __future__ import annotations

import contextlib
import os
import re
import secrets
import stat
import uuid
from pathlib import Path
from typing import Literal

from amx._log import get_logger
from amx.data.io import read_unitframe, write_unitframe
from amx.data.unitframe import UnitFrame
from amx.split.errors import VaultError

log = get_logger(__name__)

VAULT_ENV = "AMX_VAULT"
DIR_MODE = 0o700
FILE_MODE = 0o600
FOLD_NAMES = ("calib", "sealed")
KEY_BYTES = 32
KEY_FILE = "hmac.key"
FOLDS_DIR = "folds"

LOCAL_DIR_WARNING = (
    "vault mode local_dir is a convenience only. Not safe against a determined agent running "
    "as the same user (HANDOFF 8.4); use container mode for real work."
)
ROOT_NOTE = "Running as root: file modes 0700/0600 do not restrict root, so they give no isolation."
SHARED_ROOT_NOTE = (
    "The vault root {root} has mode {mode:04o}: other users can list it; run directories "
    "inside it are 0700."
)

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def default_root() -> Path:
    env = os.environ.get(VAULT_ENV)
    return Path(env).expanduser() if env else Path.home() / ".amx" / "vault"


def _check_name(kind: str, name: str) -> str:
    if not _NAME.match(name) or name in (".", ".."):
        raise VaultError(f"invalid {kind} {name!r}: use letters, digits, '.', '_' or '-'")
    return name


def _ensure_private_dir(path: Path) -> Path:
    """Create ``path`` with mode 0700, or force an existing one to 0700.

    For the directories amx owns inside the root: run directories and their ``folds``.
    """
    path.mkdir(mode=DIR_MODE, exist_ok=True)
    if stat.S_IMODE(path.stat().st_mode) != DIR_MODE:
        os.chmod(path, DIR_MODE)
    return path


def _ensure_root(path: Path) -> Path:
    """Create the vault root with mode 0700, or accept an existing directory as it is.

    An existing root is never chmodded: it may be a shared directory such as ``/tmp``. An
    existing root that is wider than 0700 and not owned by the current user is refused, since
    other users could create or swap run directories in it. Missing parents are created with
    the default mode.
    """
    try:
        path.mkdir(mode=DIR_MODE, parents=True)
    except FileExistsError:
        pass
    else:
        os.chmod(path, DIR_MODE)  # the umask may have removed owner bits
        return path
    if not path.is_dir():
        raise VaultError(f"vault root {path} exists but is not a directory")
    st = path.stat()
    dir_mode = stat.S_IMODE(st.st_mode)
    geteuid = getattr(os, "geteuid", None)
    if dir_mode & 0o077 and geteuid is not None and st.st_uid != geteuid():
        raise VaultError(
            f"vault root {path} is shared (mode {dir_mode:04o}) and not owned by the current "
            f"user (owner uid {st.st_uid}); amx neither uses nor changes it. Point {VAULT_ENV} "
            "at a private directory, or at a path amx can create (it is created with mode 0700)"
        )
    return path


def write_private(path: Path, data: bytes) -> Path:
    """Write ``data`` to ``path`` with mode 0600 through a temporary file and a rename."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE)
    try:
        with os.fdopen(fd, "wb") as fh:
            os.fchmod(fh.fileno(), FILE_MODE)
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


class LocalVault:
    """Vault in a local directory, one owner-only subdirectory per run."""

    mode: Literal["local_dir"] = "local_dir"

    def __init__(self, root: str | Path | None = None) -> None:
        base = default_root() if root is None else Path(root).expanduser()
        self.root = base.resolve()

    def __repr__(self) -> str:
        return f"LocalVault(root={str(self.root)!r})"

    # paths -------------------------------------------------------------------------------

    def run_path(self, run_id: str, *, create: bool = True) -> Path:
        """Directory of ``run_id``; with ``create`` it is made (or forced) 0700.

        A missing root is created 0700; an existing root is left as it is, and refused when it
        is wider than 0700 and owned by another user.
        """
        path = self.root / _check_name("run id", run_id)
        if create:
            _ensure_root(self.root)
            _ensure_private_dir(path)
        return path

    def file_path(self, run_id: str, name: str) -> Path:
        """Path of a top-level file of the run (the run directory is created)."""
        return self.run_path(run_id) / _check_name("file name", name)

    def _fold_path(self, run_id: str, name: str, *, create: bool) -> Path:
        if name not in FOLD_NAMES:
            raise VaultError(f"vault folds are {FOLD_NAMES}, not {name!r}")
        folds = self.run_path(run_id, create=create) / FOLDS_DIR
        if create:
            _ensure_private_dir(folds)
        return folds / f"{name}.parquet"

    # folds -------------------------------------------------------------------------------

    def has_fold(self, run_id: str, name: str) -> bool:
        return self._fold_path(run_id, name, create=False).is_file()

    def write_fold(self, run_id: str, name: str, uf: UnitFrame, *, overwrite: bool = False) -> Path:
        """Store fold ``name`` ("calib" or "sealed") with inputs, labels and roles (mode 0600).

        Refuses to replace an existing fold unless ``overwrite``: folds are hash-locked by
        the split manifest.
        """
        path = self._fold_path(run_id, name, create=True)
        if path.exists() and not overwrite:
            raise VaultError(f"fold '{name}' of run '{run_id}' already exists in the vault")
        write_unitframe(uf, path, file_mode=FILE_MODE)
        log.debug("vault: wrote fold %s of run %s (%d units)", name, run_id, uf.n)
        return path

    def read_fold(self, run_id: str, name: str) -> UnitFrame:
        """Read fold ``name`` of ``run_id``. Warden only: never call this from agent-side code."""
        path = self._fold_path(run_id, name, create=False)
        if not path.is_file():
            raise VaultError(f"fold '{name}' of run '{run_id}' is not in the vault")
        return read_unitframe(path)

    # secrets and snapshots ---------------------------------------------------------------

    def hmac_key(self, run_id: str) -> bytes:
        """The run's 32-byte HMAC key, created once (race-safe) and then reused."""
        path = self.file_path(run_id, KEY_FILE)
        if not path.exists():
            tmp = path.with_name(f".{KEY_FILE}.{uuid.uuid4().hex}.tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE)
            try:
                with os.fdopen(fd, "wb") as fh:
                    os.fchmod(fh.fileno(), FILE_MODE)
                    fh.write(secrets.token_bytes(KEY_BYTES))
                    fh.flush()
                    os.fsync(fh.fileno())
                with contextlib.suppress(FileExistsError):
                    os.link(tmp, path)  # atomic: the first writer wins
            finally:
                tmp.unlink(missing_ok=True)
        key = path.read_bytes()
        if len(key) != KEY_BYTES:
            raise VaultError(f"HMAC key of run '{run_id}' is corrupt ({len(key)} bytes)")
        return key

    def write_text(self, run_id: str, name: str, text: str, *, overwrite: bool = True) -> Path:
        """Store a text snapshot (e.g. ``taskspec.yaml``) in the run directory (mode 0600)."""
        path = self.file_path(run_id, name)
        if path.exists() and not overwrite:
            raise VaultError(f"'{name}' of run '{run_id}' already exists in the vault")
        return write_private(path, text.encode("utf-8"))

    def read_text(self, run_id: str, name: str) -> str:
        path = self.run_path(run_id, create=False) / _check_name("file name", name)
        if not path.is_file():
            raise VaultError(f"'{name}' of run '{run_id}' is not in the vault")
        return path.read_text(encoding="utf-8")

    # reporting ---------------------------------------------------------------------------

    def isolation_warning(self) -> str | None:
        """What this vault mode does not protect against (reported by ``amx doctor``)."""
        parts = [LOCAL_DIR_WARNING]
        geteuid = getattr(os, "geteuid", None)
        if geteuid is not None and geteuid() == 0:
            parts.append(ROOT_NOTE)
        if self.root.is_dir():
            root_mode = stat.S_IMODE(self.root.stat().st_mode)
            if root_mode & 0o077:
                parts.append(SHARED_ROOT_NOTE.format(root=self.root, mode=root_mode))
        return " ".join(parts)
