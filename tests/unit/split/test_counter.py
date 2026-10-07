from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path
from typing import Any

import pytest

from amx.split import (
    CertifyBudgetExhausted,
    CertifyCounter,
    LocalVault,
    SealedTouchError,
    SealedTouchLog,
    VaultError,
)


def test_counter_refuses_call_max_plus_one(tmp_path: Path) -> None:
    vault = LocalVault(tmp_path / "vault")
    counter = CertifyCounter(vault, "r", max_calls=3)
    assert counter.used() == 0 and counter.remaining() == 3
    assert [counter.acquire(f"call {i}") for i in range(3)] == [0, 1, 2]
    assert counter.used() == 3 and counter.remaining() == 0
    with pytest.raises(CertifyBudgetExhausted, match="3 of 3"):
        counter.acquire("one too many")
    assert counter.used() == 3
    records = [
        json.loads(x) for x in (vault.root / "r" / "certify_calls.jsonl").read_text().splitlines()
    ]
    assert [r["index"] for r in records] == [0, 1, 2]
    assert records[0]["purpose"] == "call 0"


def test_counter_state_survives_reopen_and_budget_cannot_grow(tmp_path: Path) -> None:
    vault = LocalVault(tmp_path / "vault")
    assert CertifyCounter(vault, "r", 1).acquire("first") == 0
    with pytest.raises(CertifyBudgetExhausted):
        CertifyCounter(vault, "r", 1).acquire("again")
    with pytest.raises(CertifyBudgetExhausted):
        CertifyCounter(vault, "r", 5).acquire("bigger budget later")
    assert CertifyCounter(vault, "other", 1).acquire("separate run") == 0
    with pytest.raises(ValueError):
        CertifyCounter(vault, "r", 0)


def test_counter_refuses_a_corrupt_log(tmp_path: Path) -> None:
    vault = LocalVault(tmp_path / "vault")
    counter = CertifyCounter(vault, "r", 3)
    counter.acquire("ok")
    with (vault.root / "r" / "certify_calls.jsonl").open("a") as fh:
        fh.write("{truncated\n")
    with pytest.raises(VaultError, match="corrupt"):
        counter.acquire("after corruption")


def _race(root: str, start: Any, out: Any, attempts: int) -> None:
    counter = CertifyCounter(LocalVault(root), "race", max_calls=3)
    start.wait(10)
    results: list[int] = []
    for _ in range(attempts):
        try:
            results.append(counter.acquire("race"))
        except CertifyBudgetExhausted:
            results.append(-1)
    out.put(results)


def test_counter_holds_across_concurrent_processes(tmp_path: Path) -> None:
    root = str(tmp_path / "vault")
    LocalVault(root).run_path("race")
    ctx = mp.get_context("spawn")
    start = ctx.Event()
    out = ctx.Queue()
    procs = [ctx.Process(target=_race, args=(root, start, out, 3)) for _ in range(3)]
    for p in procs:
        p.start()
    start.set()
    results = [out.get(timeout=60) for _ in procs]
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    granted = sorted(i for r in results for i in r if i >= 0)
    assert granted == [0, 1, 2]
    assert sum(r.count(-1) for r in results) == 3 * 3 - 3
    assert CertifyCounter(LocalVault(root), "race", 3).used() == 3


def test_sealed_touch_rule(tmp_path: Path) -> None:
    vault = LocalVault(tmp_path / "vault")
    touches = SealedTouchLog(vault, "r")
    assert touches.record("simulate") == 0
    assert touches.record("simulate") == 0  # recorded only, never refused
    assert touches.record("final_report") == 0
    with pytest.raises(SealedTouchError, match="new-sealed"):
        touches.record("final_report")
    with pytest.raises(SealedTouchError):
        SealedTouchLog(vault, "r").record("final_report")  # state lives in the vault
    assert touches.record("final_report", new_sealed=True) == 1
    with pytest.raises(SealedTouchError):
        touches.record("final_report")
    assert touches.record("simulate") == 1
    kinds = [(t["kind"], t["release"]) for t in touches.touches()]
    assert kinds == [
        ("simulate", 0),
        ("simulate", 0),
        ("final_report", 0),
        ("final_report", 1),
        ("simulate", 1),
    ]
    with pytest.raises(ValueError, match="touch kind"):
        touches.record("peek")
    with pytest.raises(ValueError, match="new_sealed"):
        touches.record("simulate", new_sealed=True)
