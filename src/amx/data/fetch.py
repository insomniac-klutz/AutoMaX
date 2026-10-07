"""Download a file from a list of mirrors, pinned by sha256 (ROLLER step 4, OQ Q4).

Uses ``urllib.request`` only. The opener is built per call, so environment proxies
(``https_proxy``, ``no_proxy``, ...) and ``SSL_CERT_FILE`` apply as they are at call time.
A file is streamed to a temporary name next to ``dest``, hashed while it streams, and renamed
into place only when its sha256 matches; otherwise the next mirror is tried.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import re
import tempfile
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from amx._log import get_logger

log = get_logger(__name__)

CHUNK_BYTES = 1 << 20
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class FetchAttempt:
    url: str
    error: str


class FetchError(RuntimeError):
    """No mirror produced a file with the expected sha256."""

    def __init__(self, dest: Path, attempts: Sequence[FetchAttempt]) -> None:
        self.dest = dest
        self.attempts = tuple(attempts)
        lines = "\n".join(f"  - {a.url}: {a.error}" for a in self.attempts) or "  (no urls)"
        super().__init__(f"could not fetch {dest.name}; attempts:\n{lines}")


def normalize_sha256(sha256: str) -> str:
    """Lower-case hex digest; an optional ``sha256:`` prefix is accepted."""
    digest = sha256.strip().lower()
    if digest.startswith("sha256:"):
        digest = digest[len("sha256:") :]
    if not _HEX64.match(digest):
        raise ValueError(f"not a sha256 hex digest: {sha256!r}")
    return digest


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(urls: Sequence[str], sha256: str, dest: Path, timeout: float = 60) -> Path:
    """Return ``dest`` holding the file whose sha256 is ``sha256``, downloading it if needed.

    Mirrors are tried in order. An HTTP or network error, or a hash mismatch, is logged and
    the next mirror is tried. :class:`FetchError` lists every attempt when all fail.
    """
    expected = normalize_sha256(sha256)
    dest = Path(dest)
    if dest.is_file():
        if file_sha256(dest) == expected:
            log.debug("fetch: %s already present with the pinned hash", dest)
            return dest
        log.warning("fetch: %s exists with a different sha256; downloading again", dest)
    if isinstance(urls, str):
        urls = [urls]
    dest.parent.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "amx-fetch/1")]
    attempts: list[FetchAttempt] = []
    for url in urls:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".part", dir=dest.parent)
        tmp = Path(tmp_name)
        try:
            h = hashlib.sha256()
            size = 0
            with os.fdopen(fd, "wb") as out, opener.open(url, timeout=timeout) as resp:
                for chunk in iter(lambda: resp.read(CHUNK_BYTES), b""):
                    h.update(chunk)
                    out.write(chunk)
                    size += len(chunk)
            got = h.hexdigest()
            if got != expected:
                error = f"sha256 mismatch (got {got}, {size} bytes)"
                log.warning("fetch: %s: %s", url, error)
                attempts.append(FetchAttempt(url, error))
                tmp.unlink(missing_ok=True)
                continue
            os.replace(tmp, dest)
            log.info("fetch: %s -> %s (%d bytes)", url, dest, size)
            return dest
        except urllib.error.HTTPError as exc:
            error = f"HTTP {exc.code} {exc.reason}"
            exc.close()
        except urllib.error.URLError as exc:
            error = f"URL error: {exc.reason}"
        except (OSError, ValueError, http.client.HTTPException) as exc:
            error = f"{type(exc).__name__}: {exc}"
        tmp.unlink(missing_ok=True)
        log.warning("fetch: %s: %s", url, error)
        attempts.append(FetchAttempt(url, error))
    raise FetchError(dest, attempts)
