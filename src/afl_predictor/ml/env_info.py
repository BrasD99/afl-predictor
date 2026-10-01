from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any


def collect_env_info() -> dict[str, str]:
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in ("numpy", "sklearn", "joblib", "pandas"):
        try:
            module = __import__(package if package != "sklearn" else "sklearn")
            versions[package] = module.__version__
        except ImportError:
            versions[package] = "not installed"
    return versions


def save_env_info(models_dir: Path) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / "model_env.json"
    path.write_text(
        json.dumps(collect_env_info(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def load_env_info(models_dir: Path) -> dict[str, Any] | None:
    path = Path(models_dir) / "model_env.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def format_env_mismatch_hint(models_dir: Path) -> str:
    saved = load_env_info(models_dir)
    current = collect_env_info()

    lines = [
        "Модели обучены в другом окружении и не загружаются (конфликт NumPy/sklearn).",
        "",
        "Текущее окружение:",
        f"  numpy={current.get('numpy')}, sklearn={current.get('sklearn')}, python={current.get('python')}",
    ]
    if saved:
        lines.extend(
            [
                "",
                "Окружение при обучении (model_env.json):",
                f"  numpy={saved.get('numpy')}, sklearn={saved.get('sklearn')}, python={saved.get('python')}",
            ]
        )

    lines.extend(
        [
            "",
            "Решения:",
            "  1) Поставить зависимости проекта и перезапустить интерпретатор:",
            "     pip install -r requirements.txt",
            "",
            "  2) Переобучить на этой машине:",
            "     python scripts/train_model.py",
        ]
    )
    return "\n".join(lines)
