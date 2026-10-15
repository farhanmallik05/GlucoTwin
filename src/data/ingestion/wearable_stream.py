"""
src/data/ingestion/wearable_stream.py

Asynchronous real-time streaming manager that dispatches multi-modal sensor readings
to active Digital Twin instances with demo acceleration (60x) and event injection.
"""

from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from src.data.ingestion.wearable_simulator import WearableReading, WearableSimulator

logger = logging.getLogger(__name__)


class WearableStreamManager:
    """
    Asynchronous streaming manager that dispatches real-time sensor readings
    to active Digital Twin instances with demo acceleration and event injection.
    """

    def __init__(self, demo_mode: bool = True, speed_multiplier: float = 60.0):
        self.demo_mode = demo_mode
        self.speed_multiplier = speed_multiplier
        self._patient_queues: Dict[str, asyncio.Queue] = {}
        self._patient_tasks: Dict[str, asyncio.Task] = {}
        self._patient_data: Dict[str, pd.DataFrame] = {}
        self._patient_indices: Dict[str, int] = {}
        self._injected_events: Dict[str, List[dict]] = {}
        self._is_running: Dict[str, bool] = {}

    async def start_patient_stream(
        self,
        patient_id: str,
        historical_data: pd.DataFrame,
    ) -> None:
        """Initializes and begins asynchronous stream dispatch for patient."""
        if patient_id in self._patient_tasks and not self._patient_tasks[patient_id].done():
            logger.info(f"Stream already running for patient {patient_id}")
            return

        self._patient_queues[patient_id] = asyncio.Queue(maxsize=100)
        self._patient_data[patient_id] = (
            historical_data.copy().reset_index(drop=True)
            if not historical_data.empty
            else pd.DataFrame()
        )
        self._patient_indices[patient_id] = 0
        self._injected_events[patient_id] = []
        self._is_running[patient_id] = True

        task = asyncio.create_task(self._stream_loop(patient_id))
        self._patient_tasks[patient_id] = task
        logger.info(f"Started streaming task for patient {patient_id} (speed: {self.speed_multiplier}x)")

    async def _stream_loop(self, patient_id: str) -> None:
        """Background coroutine feeding readings into queue at accelerated demo intervals."""
        # In 60x demo mode, 5 min (300 sec) real time is 5.0 seconds
        tick_interval = (300.0 / self.speed_multiplier) if self.demo_mode else 300.0

        try:
            while self._is_running.get(patient_id, False):
                df = self._patient_data.get(patient_id, pd.DataFrame())
                n_rows = len(df)

                # BUG-M1-03 Guard: If DataFrame is empty, idle safely without indexing
                if n_rows == 0:
                    await asyncio.sleep(tick_interval)
                    continue

                idx = self._patient_indices[patient_id]
                if idx >= n_rows:
                    idx = 0
                    self._patient_indices[patient_id] = 0

                row = df.iloc[idx].to_dict()

                # Process any injected events
                events = self._injected_events.get(patient_id, [])
                if events:
                    ev = events.pop(0)
                    row = self._apply_event_to_row(row, ev)

                # Convert row to WearableReading
                ts = row.get("timestamp")
                if isinstance(ts, str):
                    try:
                        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except Exception:
                        ts = datetime.utcnow()
                elif isinstance(ts, pd.Timestamp):
                    ts = ts.to_pydatetime()
                elif not isinstance(ts, datetime):
                    ts = datetime.utcnow()

                reading = WearableReading(
                    patient_id=patient_id,
                    timestamp=ts,
                    glucose_mgdl=float(row.get("glucose_mgdl", 140.0)),
                    glucose_roc=float(row.get("glucose_roc", 0.0)),
                    glucose_acc=float(row.get("glucose_acc", 0.0)),
                    heart_rate_bpm=float(row.get("heart_rate_bpm", 72.0)),
                    hrv_rmssd_ms=float(row.get("hrv_rmssd_ms", 38.0)),
                    hrv_sdnn_ms=float(row.get("hrv_sdnn_ms", 45.0)),
                    hrv_lf_hf_ratio=float(row.get("hrv_lf_hf_ratio", 1.5)),
                    steps_last_5min=int(row.get("steps_last_5min", 0)),
                    steps_last_hour=int(row.get("steps_last_hour", 0)),
                    activity_type=str(row.get("activity_type", "resting")),
                    calories_5min=float(row.get("calories_5min", 0.0)),
                    sleep_stage=str(row.get("sleep_stage", "none")),
                    sleep_quality=float(row.get("sleep_quality", 0.85)),
                    data_quality=str(row.get("data_quality", "good")),
                    source=str(row.get("source", "stream")),
                )

                q = self._patient_queues[patient_id]
                # If queue is full, pop oldest
                if q.full():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await q.put(reading)

                self._patient_indices[patient_id] = idx + 1
                await asyncio.sleep(tick_interval)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in streaming loop for patient {patient_id}: {e}")

    def _apply_event_to_row(self, row: dict, event: dict) -> dict:
        """Modifies row telemetry according to injected scenario event."""
        ev_type = event.get("type")
        if ev_type == "post_meal_spike":
            row["glucose_mgdl"] = min(380.0, float(row.get("glucose_mgdl", 140.0)) + 95.0)
            row["glucose_roc"] = 2.4
            row["glucose_acc"] = 0.08
            row["heart_rate_bpm"] = float(row.get("heart_rate_bpm", 72.0)) + 8.0
        elif ev_type == "insulin_overdose":
            curr_g = float(row.get("glucose_mgdl", 140.0))
            row["glucose_mgdl"] = max(40.0, min(curr_g - 70.0, 58.0))
            row["glucose_roc"] = -3.2
            row["glucose_acc"] = -0.10
            row["hrv_rmssd_ms"] = max(10.0, float(row.get("hrv_rmssd_ms", 38.0)) * 0.55)
            row["heart_rate_bpm"] = float(row.get("heart_rate_bpm", 72.0)) + 22.0
            row["hrv_lf_hf_ratio"] = 4.5
        elif ev_type == "exercise_burst":
            row["steps_last_5min"] = 520
            row["activity_type"] = "exercise"
            row["heart_rate_bpm"] = 145.0
            row["calories_5min"] = 28.0
            row["glucose_mgdl"] = max(55.0, float(row.get("glucose_mgdl", 140.0)) - 35.0)
            row["glucose_roc"] = -1.5
        return row

    async def get_next_reading(self, patient_id: str) -> Optional[WearableReading]:
        """
        Awaits and yields the next reading for patient from queue.
        Returns None when the stream has terminated.
        """
        if patient_id not in self._patient_queues:
            if not self._is_running.get(patient_id, False):
                return None
            raise KeyError(f"No stream active for patient {patient_id}")
        q = self._patient_queues[patient_id]
        reading = await q.get()
        return reading

    async def inject_event(self, patient_id: str, event_type: str) -> None:
        """
        Injects acute glycemic/metabolic event for demo testing:
        - 'post_meal_spike'
        - 'insulin_overdose'
        - 'exercise_burst'
        """
        if patient_id not in self._injected_events:
            self._injected_events[patient_id] = []
        self._injected_events[patient_id].append({"type": event_type, "timestamp": datetime.utcnow()})
        logger.info(f"Injected event '{event_type}' for patient {patient_id}")

    def get_stream_status(self, patient_id: str) -> Dict[str, Any]:
        """Returns stream position, total points, speed multiplier, active status."""
        active = self._is_running.get(patient_id, False)
        idx = self._patient_indices.get(patient_id, 0)
        total = len(self._patient_data.get(patient_id, []))
        return {
            "patient_id": patient_id,
            "active": active,
            "current_index": idx,
            "total_points": total,
            "speed_multiplier": self.speed_multiplier,
            "demo_mode": self.demo_mode,
        }

    async def stop_patient_stream(self, patient_id: str) -> None:
        """Stops and cleans up streaming background task with sentinel notification."""
        self._is_running[patient_id] = False

        # BUG-M1-04 Fix: Put None sentinel into queue to unblock awaiting consumers
        if patient_id in self._patient_queues:
            q = self._patient_queues[patient_id]
            try:
                if q.full():
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(None)
            except Exception as e:
                logger.warning(f"Could not enqueue stop sentinel for patient {patient_id}: {e}")

        # Cancel and await background task
        if patient_id in self._patient_tasks:
            task = self._patient_tasks[patient_id]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            del self._patient_tasks[patient_id]

        if patient_id in self._patient_queues:
            del self._patient_queues[patient_id]
        logger.info(f"Stopped streaming task for patient {patient_id}")
