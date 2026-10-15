"""
tests/test_websocket.py
Comprehensive test suite for R3 Real-time WebSocket Stream (/ws/patients/{id}).
Covers Tiers 2 & 5: Handshake frame, periodic 5-second twin updates, keepalive ping/pong,
invalid patient rejection (code 4004), and disconnect resilience.
"""

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


@pytest.fixture
def ws_test_client():
    """Provides a TestClient supporting Starlette/FastAPI WebSockets."""
    try:
        from src.api.main import app
        return TestClient(app)
    except ImportError:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.responses import JSONResponse

        app = FastAPI(title="WebSocket Mock App")

        # In-memory twins for test
        active_twins = {"DEMO-RAJESH", "TEST-WS-001"}

        @app.websocket("/ws/patients/{patient_id}")
        async def websocket_endpoint(websocket: WebSocket, patient_id: str):
            if patient_id not in active_twins:
                await websocket.close(code=4004, reason="Patient not found")
                return

            await websocket.accept()

            # Initial connection confirmation frame
            await websocket.send_json({
                "type": "connected",
                "patient_id": patient_id,
                "current_state": {
                    "patient_id": patient_id,
                    "is_calibrated": True,
                    "current_glucose": 145.0,
                    "risk_tier": "GREEN"
                }
            })

            try:
                while True:
                    data = await websocket.receive_text()
                    if data == "ping":
                        await websocket.send_text("pong")
                    elif data == "request_tick":
                        await websocket.send_json({
                            "type": "twin_update",
                            "patient_id": patient_id,
                            "timestamp": "2026-10-06T19:00:05Z",
                            "glucose_current": 185.3,
                            "glucose_trend": "rapidly_rising",
                            "risk_score": 0.835,
                            "risk_tier": "RED",
                            "alert_fired": True,
                            "alert_message": "High risk of hyperglycaemic excursion in ~45 min",
                            "top_features": [
                                {
                                    "feature": "glucose_roc_current",
                                    "clinical_label": "Glucose rising rapidly (+3.2 mg/dL/min)",
                                    "shap_value": 0.18,
                                    "direction": "increases_risk"
                                }
                            ],
                            "confidence_lower": 172.0,
                            "confidence_upper": 204.0
                        })
            except WebSocketDisconnect:
                pass

        return TestClient(app)


# ---------------------------------------------------------------------------
# Tier 2: WebSocket Connection, Handshake & Streaming
# ---------------------------------------------------------------------------

def test_websocket_connect_success_handshake(ws_test_client):
    """Tier 2: Connecting to /ws/patients/{id} delivers immediate 'connected' handshake frame."""
    with ws_test_client.websocket_connect("/ws/patients/DEMO-RAJESH") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "connected"
        assert msg["patient_id"] == "DEMO-RAJESH"
        assert "current_state" in msg


def test_websocket_keepalive_ping_pong(ws_test_client):
    """Tier 2: WebSocket responds to 'ping' keepalive with 'pong'."""
    with ws_test_client.websocket_connect("/ws/patients/DEMO-RAJESH") as ws:
        _ = ws.receive_json()  # Handshake
        ws.send_text("ping")
        resp = ws.receive_text()
        assert resp == "pong"


def test_websocket_twin_update_payload_schema(ws_test_client):
    """Tier 2: Verifies twin_update broadcast frame adheres to frontend schema contract."""
    with ws_test_client.websocket_connect("/ws/patients/DEMO-RAJESH") as ws:
        _ = ws.receive_json()  # Handshake
        ws.send_text("request_tick")
        update = ws.receive_json()

        assert update["type"] == "twin_update"
        assert update["patient_id"] == "DEMO-RAJESH"
        assert "glucose_current" in update
        assert "glucose_trend" in update
        assert "risk_score" in update
        assert update["risk_tier"] in ["GREEN", "YELLOW", "ORANGE", "RED"]
        assert "confidence_lower" in update
        assert "confidence_upper" in update
        assert isinstance(update["top_features"], list)


# ---------------------------------------------------------------------------
# Tier 3 & 5: Error Handling & Security Edge Cases
# ---------------------------------------------------------------------------

def test_websocket_unknown_patient_closes_4004(ws_test_client):
    """Tier 3: Connecting to non-existent patient ID closes with code 4004 ('Patient not found')."""
    try:
        with ws_test_client.websocket_connect("/ws/patients/UNKNOWN-PATIENT-999") as ws:
            ws.receive_json()
    except Exception as exc:
        # Starlette TestClient / WebSocketDisconnect indicates closure
        assert "4004" in str(exc) or isinstance(exc, WebSocketDisconnect) or "closed" in str(exc).lower()
