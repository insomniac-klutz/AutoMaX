"""Canonical hashing of specs and JSON-like payloads."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, NaN rejected."""
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def content_hash(obj: Any) -> str:
    """Hash of the canonical JSON form; independent of key order."""
    return sha256_hex(canonical_json(obj))
