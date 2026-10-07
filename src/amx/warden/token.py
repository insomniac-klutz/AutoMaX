"""Freeze tokens: the human-held capability that unlocks warden commands (HANDOFF 8.4).

The token is printed once to the human who issues it; only its sha256 is stored in the vault.
It must never be placed in an agent's environment. In ``local_dir`` vault mode the token is a
procedural gate, not a security boundary: any process running as the same user can issue one.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets

from amx.split.errors import VaultError
from amx.split.vault import LocalVault

TOKEN_FILE = "freeze_token.sha256"
ENV_VAR = "AMX_FREEZE_TOKEN"


class TokenError(PermissionError):
    """A warden command was called without a valid freeze token."""


def issue_token(vault: LocalVault, run_id: str, *, rotate: bool = False) -> str:
    """Issue the run's freeze token. Re-issuing needs ``rotate=True`` (it voids the old token)."""
    if not rotate:
        try:
            vault.read_text(run_id, TOKEN_FILE)
        except (FileNotFoundError, VaultError):
            pass
        else:
            raise TokenError("a freeze token already exists for this run; rotating needs --rotate")
    token = secrets.token_urlsafe(32)
    vault.write_text(run_id, TOKEN_FILE, hashlib.sha256(token.encode()).hexdigest())
    return token


def resolve_token(explicit: str | None) -> str | None:
    return explicit if explicit else os.environ.get(ENV_VAR)


def verify_token(vault: LocalVault, run_id: str, token: str | None) -> None:
    if not token:
        raise TokenError("this is a warden command: pass the freeze token issued by a human")
    try:
        stored = vault.read_text(run_id, TOKEN_FILE).strip()
    except (FileNotFoundError, VaultError) as exc:
        raise TokenError("no freeze token has been issued for this run") from exc
    given = hashlib.sha256(token.encode()).hexdigest()
    if not hmac.compare_digest(stored, given):
        raise TokenError("freeze token does not match")
