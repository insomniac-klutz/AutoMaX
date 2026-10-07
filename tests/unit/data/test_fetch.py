"""fetch() against a local threaded http.server (no network)."""

from __future__ import annotations

import functools
import hashlib
import threading
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from amx.data import FetchError, fetch

PAYLOAD = b"id,x,y\n" + b"".join(f"u{i},{i * 0.5},{i % 2}\n".encode() for i in range(2000))
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[str, Path]]:
    """Serve a tmp dir on 127.0.0.1; returns (base url, served dir)."""
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(var, "127.0.0.1,localhost")
    root = tmp_path / "www"
    root.mkdir()
    (root / "good.csv").write_bytes(PAYLOAD)
    (root / "tampered.csv").write_bytes(PAYLOAD.replace(b"u7,", b"u8,"))
    handler = functools.partial(_QuietHandler, directory=str(root))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(
        target=httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
    )
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", root
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def leftovers(dest: Path) -> list[Path]:
    return [p for p in dest.parent.iterdir() if p.name.endswith(".part")]


def test_fetch_falls_back_after_404(server: tuple[str, Path], tmp_path: Path) -> None:
    base, _ = server
    dest = tmp_path / "out" / "data.csv"
    got = fetch([f"{base}/missing.csv", f"{base}/good.csv"], DIGEST, dest, timeout=10)
    assert got == dest and dest.read_bytes() == PAYLOAD
    assert leftovers(dest) == []


def test_fetch_falls_back_after_hash_mismatch(server: tuple[str, Path], tmp_path: Path) -> None:
    base, _ = server
    dest = tmp_path / "data.csv"
    fetch([f"{base}/tampered.csv", f"{base}/good.csv"], "sha256:" + DIGEST.upper(), dest)
    assert dest.read_bytes() == PAYLOAD
    assert leftovers(dest) == []


def test_fetch_early_return_when_present(server: tuple[str, Path], tmp_path: Path) -> None:
    base, root = server
    dest = tmp_path / "data.csv"
    dest.write_bytes(PAYLOAD)
    mtime = dest.stat().st_mtime_ns
    (root / "good.csv").unlink()  # the only mirror is gone: any download would fail
    assert fetch([f"{base}/good.csv"], DIGEST, dest) == dest
    assert dest.stat().st_mtime_ns == mtime


def test_fetch_replaces_a_stale_file(server: tuple[str, Path], tmp_path: Path) -> None:
    base, _ = server
    dest = tmp_path / "data.csv"
    dest.write_bytes(b"stale")
    fetch([f"{base}/good.csv"], DIGEST, dest)
    assert dest.read_bytes() == PAYLOAD


def test_fetch_error_lists_every_attempt(server: tuple[str, Path], tmp_path: Path) -> None:
    base, _ = server
    dest = tmp_path / "data.csv"
    urls = [f"{base}/missing.csv", f"{base}/tampered.csv", "http://127.0.0.1:9/refused"]
    with pytest.raises(FetchError) as info:
        fetch(urls, DIGEST, dest, timeout=5)
    err = info.value
    assert [a.url for a in err.attempts] == urls
    assert "404" in err.attempts[0].error
    assert "sha256 mismatch" in err.attempts[1].error
    for url in urls:
        assert url in str(err)
    assert not dest.exists()
    assert leftovers(dest) == []


def test_fetch_rejects_a_malformed_digest(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sha256"):
        fetch(["http://127.0.0.1:9/x"], "md5:abc", tmp_path / "x")
