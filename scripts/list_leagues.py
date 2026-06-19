#!/usr/bin/env python3
"""Запуск без pip install -e . — добавляет src/ в PYTHONPATH."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

runpy.run_module("afl_predictor.scripts.list_leagues", run_name="__main__")
