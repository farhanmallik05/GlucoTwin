"""
src/api/middleware/auth.py
API Key Authentication Dependency.
Enforces security and DPDP Act 2023 access controls on protected endpoints.
"""

import os
from fastapi import Header, HTTPException, status

def get_valid_api_key() -> str:
    """Retrieve expected API key from environment with secure fallback."""
    return os.environ.get("GLUCOTWIN_API_KEY", os.environ.get("API_KEY", "demo-glucotwin-2026"))


VALID_API_KEY = get_valid_api_key()


async def verify_api_key(x_api_key: str = Header(None)) -> str:
    """Validate X-API-Key header against configured secret. Raises 401 if missing or invalid."""
    expected_key = get_valid_api_key()
    if not x_api_key or x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide a valid 'X-API-Key' header.",
        )
    return x_api_key

