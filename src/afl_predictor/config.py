from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class ApiConfig:
    base_url: str
    timeout_seconds: int
    page_size: int


@dataclass(frozen=True)
class DatabaseConfig:
    path: Path


@dataclass(frozen=True)
class LeagueConfig:
    id: int
    name: str | None = None


@dataclass(frozen=True)
class FetchConfig:
    finished_state_code: int


@dataclass(frozen=True)
class TrainingConfig:
    form_window: int
    h2h_window: int
    test_fraction: float
    random_state: int
    models_dir: Path
    reports_dir: Path
    generate_plots: bool
    optuna_trials: int
    optuna_timeout_seconds: int


@dataclass(frozen=True)
class AppConfig:
    api: ApiConfig
    database: DatabaseConfig
    league: LeagueConfig
    fetch: FetchConfig
    training: TrainingConfig


def _resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parse_league(raw: dict[str, Any]) -> LeagueConfig:
    if "league" in raw:
        item = raw["league"]
        if not item or "id" not in item:
            raise ValueError("В config.yaml задайте league.id")
        return LeagueConfig(id=int(item["id"]), name=item.get("name"))

    legacy = raw.get("leagues", [])
    if len(legacy) > 1:
        raise ValueError(
            "В config.yaml допускается только одна лига. "
            "Замените leagues: [...] на league: { id: ..., name: ... }"
        )
    if not legacy:
        raise ValueError("В config.yaml задайте league.id (одна лига)")
    item = legacy[0]
    return LeagueConfig(id=int(item["id"]), name=item.get("name"))


def load_config(path: Path | str | None = None) -> AppConfig:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    training_raw = raw.get("training", {})
    test_fraction = float(
        training_raw.get("test_fraction", training_raw.get("test_season_fraction", 0.2))
    )

    return AppConfig(
        api=ApiConfig(
            base_url=raw["api"]["base_url"],
            timeout_seconds=int(raw["api"].get("timeout_seconds", 30)),
            page_size=int(raw["api"].get("page_size", 50)),
        ),
        database=DatabaseConfig(path=_resolve_path(raw["database"]["path"])),
        league=_parse_league(raw),
        fetch=FetchConfig(
            finished_state_code=int(raw["fetch"].get("finished_state_code", 4)),
        ),
        training=TrainingConfig(
            form_window=int(training_raw.get("form_window", 5)),
            h2h_window=int(training_raw.get("h2h_window", 5)),
            test_fraction=test_fraction,
            random_state=int(training_raw.get("random_state", 42)),
            models_dir=_resolve_path(training_raw.get("models_dir", "data/models")),
            reports_dir=_resolve_path(training_raw.get("reports_dir", "data/reports")),
            generate_plots=bool(training_raw.get("generate_plots", True)),
            optuna_trials=int(training_raw.get("optuna_trials", 30)),
            optuna_timeout_seconds=int(training_raw.get("optuna_timeout_seconds", 600)),
        ),
    )
