#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import argparse
import logging
import sys
from pathlib import Path

def _bootstrap() -> None:
    path = Path(__file__).resolve().parent / "_bootstrap.py"
    spec = importlib.util.spec_from_file_location("_bootstrap", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

_bootstrap()

from afl_predictor.config import load_config
from afl_predictor.db import MatchRepository, get_session_factory, init_db
from afl_predictor.features import build_training_dataset
from afl_predictor.ml import train_and_save
from afl_predictor.ml.constants import FEATURE_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Обучение модели для одной лиги AFL")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Путь к config.yaml (по умолчанию config/config.yaml)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Не генерировать PNG-отчёты",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    league = config.league
    db_path = config.database.path

    if not db_path.exists():
        logger.error("База данных не найдена: %s. Сначала запустите fetch_matches.", db_path)
        return 1

    init_db(str(db_path))
    session_factory = get_session_factory(str(db_path))

    with session_factory() as session:
        repo = MatchRepository(session)
        matches = repo.list_finished_matches(
            config.fetch.finished_state_code,
            league_id=league.id,
        )
        season_championship_map = repo.get_season_championship_map()

    logger.info("Лига: %s (%s)", league.name or league.id, league.id)
    logger.info("Завершённых матчей в лиге: %s", len(matches))
    if not matches:
        logger.error("Нет данных для обучения. Запустите fetch_matches для этой лиги.")
        return 1

    logger.info("Построение признаков для %s матчей...", len(matches))
    dataset = build_training_dataset(
        matches,
        season_championship_map=season_championship_map,
        form_window=config.training.form_window,
        h2h_window=config.training.h2h_window,
    )
    logger.info("Признаки построены: %s строк, %s признаков", len(dataset), len(FEATURE_COLUMNS))
    logger.info("Архитектура: Poisson (LightGBM objective=poisson, 2 регрессора)")

    if config.training.optuna_trials > 0:
        logger.info("Подбор гиперпараметров Optuna: %s trials", config.training.optuna_trials)

    generate_plots = config.training.generate_plots and not args.no_plots

    report = train_and_save(
        dataset=dataset,
        models_dir=config.training.models_dir,
        league_id=league.id,
        league_name=league.name,
        random_state=config.training.random_state,
        test_fraction=config.training.test_fraction,
        optuna_trials=config.training.optuna_trials,
        optuna_timeout_seconds=config.training.optuna_timeout_seconds,
        reports_dir=config.training.reports_dir,
        generate_plots=generate_plots,
    )

    logger.info("Split: %s (train=%s, test=%s)", report.split_type, report.train_size, report.test_size)
    logger.info("Точность исхода (H/D/A): %.1f%%", report.outcome_accuracy * 100)
    logger.info("MAE голов хозяев: %.2f", report.home_score_mae)
    logger.info("MAE голов гостей: %.2f", report.away_score_mae)
    logger.info("Модели сохранены в: %s", report.models_dir)
    if report.report_paths:
        plot_paths = [p for p in report.report_paths if p.suffix == ".png"]
        logger.info("Отчёты (%s PNG): %s", len(plot_paths), report.reports_dir)
        for path in plot_paths:
            logger.info("  %s", path.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
