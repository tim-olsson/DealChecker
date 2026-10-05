"""Shared test setup: import the app from src/ with telemetry and cloud services off."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Unit tests are offline: no Cloud Logging/Trace, no Model Armor plugin.
os.environ["LOG_TO_CLOUD"] = "false"
os.environ["TRACE_TO_CLOUD"] = "false"
os.environ["LOG_LEVEL"] = "ERROR"
os.environ.pop("MODEL_ARMOR_PROMPT_TEMPLATE", None)
os.environ.pop("MODEL_ARMOR_RESPONSE_TEMPLATE", None)
