"""
src/data/synthetic/wearable_simulator.py

Re-exports WearableSimulator and WearableReading from src.data.ingestion.wearable_simulator
for flexible package imports across test suites and digital twin modules.
"""

from src.data.ingestion.wearable_simulator import WearableSimulator, WearableReading

__all__ = ["WearableSimulator", "WearableReading"]
