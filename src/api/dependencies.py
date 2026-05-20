"""Shared FastAPI dependencies (auth, etc.)."""
from __future__ import annotations

import os
import secrets

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

_security = HTTPBasic()
_API_USER = os.environ.get("API_USERNAME", "admin")
_API_PASS = os.environ.get("API_PASSWORD", "mdp123")


def require_auth(credentials: HTTPBasicCredentials = Depends(_security)) -> None:
    """HTTP Basic auth guard — raises 401 on bad credentials."""
    ok = (
        secrets.compare_digest(credentials.username, _API_USER)
        and secrets.compare_digest(credentials.password, _API_PASS)
    )
    if not ok:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
