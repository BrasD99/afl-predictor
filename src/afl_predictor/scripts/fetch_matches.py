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

from afl_predictor.api.footballista import FootballistaClient
from afl_predictor.config import load_config
from afl_predictor.db import MatchRepository, get_session_factory, init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Загрузка матчей AFL из Footballista API")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Путь к config.yaml (по умолчанию config/config.yaml)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    league_cfg = config.league

    db_path = config.database.path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    init_db(str(db_path))

    client = FootballistaClient(
        base_url=config.api.base_url,
        timeout_seconds=config.api.timeout_seconds,
        page_size=config.api.page_size,
    )
    session_factory = get_session_factory(str(db_path))

    total_inserted = 0
    total_skipped = 0
    total_ignored = 0

    with session_factory() as session:
        repo = MatchRepository(session)

        league_id = league_cfg.id
        logger.info("Загрузка лиги %s (%s)", league_id, league_cfg.name or "")

        try:
            league = client.get_league_info(league_id)
        except Exception:
            logger.exception("Не удалось получить лигу %s", league_id)
            return 1

        repo.upsert_league(league)
        session.commit()

        for championship in league.championships:
            repo.upsert_championship(league.id, championship.id, championship.name)

            for season in championship.seasons:
                logger.info(
                    "  Сезон %s (%s) — championship %s",
                    season.id,
                    season.name,
                    championship.name,
                )
                try:
                    calendar = client.get_season_calendar(season.id)
                except Exception:
                    logger.exception("Не удалось получить календарь сезона %s", season.id)
                    continue

                inserted, skipped, ignored = repo.import_season_calendar(calendar, league.id)
                session.commit()
                total_inserted += inserted
                total_skipped += skipped
                total_ignored += ignored
                logger.info(
                    "    Добавлено: %s, пропущено (уже в БД): %s, без даты: %s",
                    inserted,
                    skipped,
                    ignored,
                )

    logger.info(
        "Готово. Всего добавлено: %s, пропущено: %s, без даты: %s",
        total_inserted,
        total_skipped,
        total_ignored,
    )
    logger.info("База данных: %s", db_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
