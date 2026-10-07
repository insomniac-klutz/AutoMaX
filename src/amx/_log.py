"""Library logging. Library code logs; it never prints."""

from __future__ import annotations

import json
import logging
from typing import Any

_ROOT = "amx"


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for supervisor ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "amx", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the ``amx`` namespace."""
    if not name.startswith(_ROOT):
        name = f"{_ROOT}.{name}"
    return logging.getLogger(name)


def configure(level: int = logging.INFO, json_lines: bool = False) -> None:
    """Configure the ``amx`` logger for CLI use. Idempotent."""
    root = logging.getLogger(_ROOT)
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter() if json_lines else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(handler)
    root.propagate = False
