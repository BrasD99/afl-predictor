"""Добавляет src/ в sys.path — для прямого запуска скриптов из IDE."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parents[2])
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
