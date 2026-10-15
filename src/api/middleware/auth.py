"""
src/api/middleware/auth.py
API Key Authentication Dependency.
Enforces security and DPDP Act 2023 access controls on protected endpoints.
"""

from fastapi import Header, HTTPException, status

VALID_API_KEY = "demo-glucotwin-2026"


async def verify_api_key(x_api_key: str = Header(None)) -> str:
    """Validate X-API-Key header. Raises 401 if missing or invalid."""
    if not x_api_key or x_api_key != VALID_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Please provide 'X-API-Key: demo-glucotwin-2026'",
        )
    return x_api_key
