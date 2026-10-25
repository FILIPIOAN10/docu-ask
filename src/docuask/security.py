"""API-key authentication as a FastAPI dependency (the Python twin of a Spring Security filter)."""

import hmac
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from docuask.config import Settings, get_settings


def key_is_valid(supplied: str | None, keys: list[str]) -> bool:
    """One rule for every door (REST header, MCP bearer token)."""
    if not supplied:
        return False
    supplied_bytes = supplied.encode()
    # compare_digest takes the same time whether the first or the last character differs, so response timing
    # cannot be used to guess a key one character at a time. Check ALL keys, do not stop at the first match.
    matches = [hmac.compare_digest(supplied_bytes, key.encode()) for key in keys]
    return any(matches)


def require_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    if key_is_valid(x_api_key, settings.keys):
        return
    raise HTTPException(
        status.HTTP_401_UNAUTHORIZED, "Missing or invalid API key", headers={"WWW-Authenticate": "ApiKey"}
    )
