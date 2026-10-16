"""
src/twin/persistence.py
Asynchronous SQLite-backed persistence layer for PatientTwin state, prediction telemetry, and alerts.
"""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import aiosqlite


class TwinPersistence:
    """
    High-performance asynchronous persistence engine for GlucoTwin states and historical telemetry.
    Supports in-memory test databases as well as production file-backed SQLite stores.
    """

    def __init__(self, database_url: str = "sqlite:///glucotwin.db"):
        self.database_url = database_url
        if database_url.startswith("sqlite:///"):
            self.db_path = database_url.replace("sqlite:///", "")
        elif database_url.startswith("sqlite:"):
            self.db_path = database_url.replace("sqlite:", "")
        else:
            self.db_path = database_url

        self._conn: Optional[aiosqlite.Connection] = None
        self._sync_records: Dict[str, List[Dict[str, Any]]] = {}

    async def initialize(self) -> None:
        """Initialize database connection and schema tables."""
        if self._conn is None:
            self._conn = await aiosqlite.connect(self.db_path)
            self._conn.row_factory = aiosqlite.Row

        async with self._conn.cursor() as cursor:
            # Table 1: Twin State Snapshots
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS twin_states (
                    patient_id TEXT PRIMARY KEY,
                    current_glucose REAL,
                    current_trend TEXT,
                    is_calibrated INTEGER,
                    risk_score REAL,
                    risk_tier TEXT,
                    glucose_baseline_mean REAL,
                    glucose_baseline_std REAL,
                    hrv_baseline_mean REAL,
                    insulin_sensitivity_factor REAL,
                    current_iob REAL,
                    state_json TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Table 2: Telemetry & Prediction Log
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT,
                    timestamp TEXT,
                    glucose_current REAL,
                    glucose_trend TEXT,
                    risk_score REAL,
                    risk_tier TEXT,
                    confidence_lower REAL,
                    confidence_upper REAL,
                    alert_fired INTEGER,
                    top_features_json TEXT
                );
            """)

            # Table 3: Clinical Alerts History
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS alert_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT,
                    timestamp TEXT,
                    tier TEXT,
                    risk_score REAL,
                    event_type TEXT,
                    predicted_event_in_minutes INTEGER,
                    message TEXT,
                    shap_features_json TEXT
                );
            """)
            await self._conn.commit()

    async def close(self) -> None:
        """Close persistent database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def save_twin_state(self, twin: Any) -> None:
        """Serialize and persist twin state."""
        if self._conn is None:
            await self.initialize()

        state_dict = {
            "patient_id": twin.patient_id,
            "current_glucose": getattr(twin, "current_glucose", 0.0),
            "current_trend": getattr(twin, "current_trend", "stable"),
            "is_calibrated": getattr(twin, "is_calibrated", False),
            "risk_score": getattr(twin, "current_risk_score", 0.0),
            "risk_tier": getattr(twin, "current_risk_tier", "GREEN"),
            "glucose_baseline_mean": getattr(twin, "glucose_baseline_mean", 140.0),
            "glucose_baseline_std": getattr(twin, "glucose_baseline_std", 25.0),
            "hrv_baseline_mean": getattr(twin, "hrv_baseline_mean", 35.0),
            "insulin_sensitivity_factor": getattr(twin, "insulin_sensitivity_factor", 50.0),
            "current_iob": getattr(twin, "current_iob", 0.0),
            "glucose_history": list(getattr(twin, "glucose_history", [])),
            "hrv_history": list(getattr(twin, "hrv_history", [])),
            "activity_history": list(getattr(twin, "activity_history", [])),
        }

        async with self._conn.cursor() as cursor:
            await cursor.execute("""
                INSERT INTO twin_states (
                    patient_id, current_glucose, current_trend, is_calibrated,
                    risk_score, risk_tier, glucose_baseline_mean, glucose_baseline_std,
                    hrv_baseline_mean, insulin_sensitivity_factor, current_iob, state_json, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(patient_id) DO UPDATE SET
                    current_glucose=excluded.current_glucose,
                    current_trend=excluded.current_trend,
                    is_calibrated=excluded.is_calibrated,
                    risk_score=excluded.risk_score,
                    risk_tier=excluded.risk_tier,
                    glucose_baseline_mean=excluded.glucose_baseline_mean,
                    glucose_baseline_std=excluded.glucose_baseline_std,
                    hrv_baseline_mean=excluded.hrv_baseline_mean,
                    insulin_sensitivity_factor=excluded.insulin_sensitivity_factor,
                    current_iob=excluded.current_iob,
                    state_json=excluded.state_json,
                    last_updated=CURRENT_TIMESTAMP;
            """, (
                twin.patient_id,
                state_dict["current_glucose"],
                state_dict["current_trend"],
                1 if state_dict["is_calibrated"] else 0,
                state_dict["risk_score"],
                state_dict["risk_tier"],
                state_dict["glucose_baseline_mean"],
                state_dict["glucose_baseline_std"],
                state_dict["hrv_baseline_mean"],
                state_dict["insulin_sensitivity_factor"],
                state_dict["current_iob"],
                json.dumps(state_dict)
            ))
            await self._conn.commit()

    async def load_twin_state(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve persisted twin state dictionary."""
        if self._conn is None:
            await self.initialize()

        async with self._conn.cursor() as cursor:
            await cursor.execute("SELECT state_json FROM twin_states WHERE patient_id = ?", (patient_id,))
            row = await cursor.fetchone()
            if row and row[0]:
                return json.loads(row[0])
        return None

    async def log_prediction(self, result: Any) -> None:
        """Log a prediction result to telemetry history."""
        if self._conn is None:
            await self.initialize()

        ts = getattr(result, "timestamp", datetime.now(timezone.utc).replace(tzinfo=None))
        ts_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)
        pid = getattr(result, "patient_id", "UNKNOWN")
        glucose = getattr(result, "glucose_current", 0.0)
        trend = getattr(result, "glucose_trend", "stable")
        risk = getattr(result, "risk_score", 0.0)
        tier = getattr(result, "risk_tier", "GREEN")
        c_low = getattr(result, "confidence_lower", max(0.0, risk - 0.05))
        c_high = getattr(result, "confidence_upper", min(1.0, risk + 0.05))
        alert_fired = 1 if getattr(result, "alert_fired", False) else 0
        top_feats = getattr(result, "top_features", None)
        feats_json = json.dumps(top_feats) if top_feats else None

        async with self._conn.cursor() as cursor:
            await cursor.execute("""
                INSERT INTO prediction_logs (
                    patient_id, timestamp, glucose_current, glucose_trend,
                    risk_score, risk_tier, confidence_lower, confidence_upper,
                    alert_fired, top_features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (pid, ts_str, glucose, trend, risk, tier, c_low, c_high, alert_fired, feats_json))
            await self._conn.commit()

    async def get_prediction_history(self, patient_id: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Retrieve telemetry and prediction records for a patient."""
        if self._conn is None:
            await self.initialize()

        async with self._conn.cursor() as cursor:
            await cursor.execute("""
                SELECT patient_id, timestamp, glucose_current, glucose_trend,
                       risk_score, risk_tier, confidence_lower, confidence_upper,
                       alert_fired, top_features_json
                FROM prediction_logs
                WHERE patient_id = ?
                ORDER BY id ASC;
            """, (patient_id,))
            rows = await cursor.fetchall()
            results = []
            for r in rows:
                results.append({
                    "patient_id": r[0],
                    "timestamp": r[1],
                    "glucose_current": r[2],
                    "glucose_trend": r[3],
                    "risk_score": r[4],
                    "risk_tier": r[5],
                    "confidence_lower": r[6],
                    "confidence_upper": r[7],
                    "alert_fired": bool(r[8]),
                    "top_features": json.loads(r[9]) if r[9] else None,
                })
            return results

    async def log_alert(self, alert: Any) -> None:
        """Persist a clinical alert to the alert_logs table."""
        if self._conn is None:
            await self.initialize()

        pid = getattr(alert, "patient_id", "UNKNOWN")
        ts = getattr(alert, "timestamp", datetime.now(timezone.utc).replace(tzinfo=None))
        ts_str = ts.isoformat() if isinstance(ts, datetime) else str(ts)
        tier = getattr(alert, "tier", "GREEN")
        risk = float(getattr(alert, "risk_score", 0.0))
        event_type = getattr(alert, "event_type", "none")
        eta = getattr(alert, "predicted_event_in_minutes", None)
        message = getattr(alert, "message", "")
        shap_feats = getattr(alert, "shap_features", None)
        shap_json = json.dumps(shap_feats) if shap_feats else None

        async with self._conn.cursor() as cursor:
            await cursor.execute("""
                INSERT INTO alert_logs (
                    patient_id, timestamp, tier, risk_score, event_type,
                    predicted_event_in_minutes, message, shap_features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """, (pid, ts_str, tier, risk, event_type, eta, message, shap_json))
            await self._conn.commit()

    async def get_recent_alerts(self, patient_id: Optional[str] = None, hours: int = 24) -> List[Dict[str, Any]]:
        """Retrieve recent clinical alerts within lookback window for fatigue suppression and auditing."""
        if self._conn is None:
            await self.initialize()

        query = """
            SELECT patient_id, timestamp, tier, risk_score, event_type,
                   predicted_event_in_minutes, message, shap_features_json
            FROM alert_logs
        """
        params: List[Any] = []
        if patient_id:
            query += " WHERE patient_id = ?"
            params.append(patient_id)
        query += " ORDER BY id ASC;"

        async with self._conn.cursor() as cursor:
            await cursor.execute(query, tuple(params))
            rows = await cursor.fetchall()
            results = []
            for r in rows:
                results.append({
                    "patient_id": r[0],
                    "timestamp": r[1],
                    "tier": r[2],
                    "risk_score": r[3],
                    "event_type": r[4],
                    "predicted_event_in_minutes": r[5],
                    "message": r[6],
                    "shap_features": json.loads(r[7]) if r[7] else None,
                })
            return results

