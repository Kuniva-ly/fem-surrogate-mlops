"""Shared FastAPI dependencies (auth, etc.)."""
from __future__ import annotations

import os
import secrets

from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials

_security = HTTPBasic()


def require_auth(credentials: HTTPBasicCredentials = Depends(_security)) -> None:
    """HTTP Basic auth guard — raises 401 on bad credentials.

    Credentials are read from API_USERNAME / API_PASSWORD environment variables.
    Both variables are mandatory; the server raises RuntimeError at the first
    authenticated request if either is absent.
    """
    api_user = os.environ.get("API_USERNAME")
    api_pass = os.environ.get("API_PASSWORD")

    if not api_user or not api_pass:
        raise RuntimeError(
            "API_USERNAME and API_PASSWORD environment variables must be set. "
            "Add them to your .env file (see .env.example)."
        )

    ok = (
        secrets.compare_digest(credentials.username, api_user)
        and secrets.compare_digest(credentials.password, api_pass)
    )
    if not ok:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
