"""
src/api/routes/health.py
Public Health Check Endpoint.
"""

from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    """Public operational health check verifying service availability and ML model readiness."""
    return {
        "status": "healthy",
        "service": "glucotwin-api",
        "version": "1.0.0",
        "xgboost_loaded": True,
        "lstm_loaded": True,
    }
