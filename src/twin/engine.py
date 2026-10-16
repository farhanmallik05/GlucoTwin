"""
src/twin/engine.py
Multi-Patient Twin Coordinator and Lifecycle Manager.
Coordinates concurrent PatientTwin instances, telemetry ingestion, and alert dispatch.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional
from src.twin.patient_twin import PatientTwin, PredictionResult
from src.twin.alert_system import Alert


class TwinNotFoundError(Exception):
    """Raised when an operation targets a non-existent patient twin."""
    pass


class TwinEngine:
    """
    Multi-Patient Digital Twin Orchestrator.
    Manages isolated PatientTwin state machines for multiple concurrent patients,
    dispatches incoming sensor frames, and executes alert callback pipelines.
    """

    def __init__(
        self,
        xgb_model: Any = None,
        lstm_model: Any = None,
        ensemble_model: Any = None,
        shap_explainer: Any = None
    ):
        self.twins: Dict[str, PatientTwin] = {}
        if xgb_model is None or lstm_model is None or ensemble_model is None:
            try:
                from src.models.xgboost_model import XGBoostAEPredictor
                from src.models.lstm_model import LSTMAEPredictor
                from src.models.ensemble import EnsemblePredictor
                from src.explainability.shap_explainer import SHAPExplainer

                xgb = xgb_model or XGBoostAEPredictor()
                lstm = lstm_model or LSTMAEPredictor()
                ens = ensemble_model or EnsemblePredictor(xgb_model=xgb, lstm_model=lstm)
                shap = shap_explainer or SHAPExplainer(model=xgb)
                self.xgb_model = xgb
                self.lstm_model = lstm
                self.ensemble_model = ens
                self.shap_explainer = shap
            except Exception:
                self.xgb_model = xgb_model
                self.lstm_model = lstm
                self.ensemble_model = ensemble_model
                self.shap_explainer = shap_explainer
        else:
            self.xgb_model = xgb_model
            self.lstm_model = lstm
            self.ensemble_model = ensemble_model
            self.shap_explainer = shap_explainer

        self.alert_callbacks: List[Callable[[Alert], Any]] = []

    def get_or_create_twin(self, patient_id: str, ehr_features: Dict[str, Any]) -> PatientTwin:
        """Retrieve existing twin or instantiate a new one."""
        if patient_id in self.twins:
            return self.twins[patient_id]
        return self.create_twin(patient_id, ehr_features)

    def create_twin(self, patient_id: str, ehr_features: Dict[str, Any]) -> PatientTwin:
        """Instantiate a new PatientTwin and attach engine models."""
        twin = PatientTwin(patient_id, ehr_features)
        twin._xgb_model = self.xgb_model
        twin._lstm_model = self.lstm_model
        twin._ensemble_model = self.ensemble_model
        twin._shap_explainer = self.shap_explainer
        self.twins[patient_id] = twin
        return twin

    def get_twin(self, patient_id: str) -> Optional[PatientTwin]:
        """Fetch active twin by patient ID."""
        return self.twins.get(patient_id)

    def remove_twin(self, patient_id: str) -> bool:
        """Decommission and remove twin from memory."""
        return bool(self.twins.pop(patient_id, None))

    def register_alert_callback(self, callback: Callable[[Alert], Any]) -> None:
        """Register listener invoked when high-priority alerts fire."""
        self.alert_callbacks.append(callback)

    async def _trigger_alert_callbacks(self, alert: Alert) -> None:
        """Notify all registered alert listeners asynchronously."""
        for cb in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(alert)
                else:
                    cb(alert)
            except Exception as e:
                pass

    async def process_reading(self, reading: Any) -> PredictionResult:
        """Route incoming wearable telemetry to the appropriate patient twin."""
        patient_id = getattr(reading, "patient_id", None)
        if not patient_id or patient_id not in self.twins:
            raise TwinNotFoundError(f"Twin not found for patient '{patient_id}'")

        twin = self.twins[patient_id]
        result = twin.update(reading)

        if result.alert_fired and result.alert:
            await self._trigger_alert_callbacks(result.alert)

        return result

    async def process_batch(self, readings: List[Any]) -> List[PredictionResult]:
        """Process batch of readings concurrently."""
        tasks = [self.process_reading(r) for r in readings]
        return await asyncio.gather(*tasks)

    def get_all_twin_summaries(self) -> List[Dict[str, Any]]:
        """Return state summaries for all registered patient twins."""
        return [twin.get_summary() for twin in self.twins.values()]


# Global TwinEngine singleton for application lifecycle
_global_engine: Optional[TwinEngine] = None


def get_twin_engine() -> TwinEngine:
    """Retrieve or lazily initialize the application TwinEngine singleton."""
    global _global_engine
    if _global_engine is None:
        _global_engine = TwinEngine()
    return _global_engine
