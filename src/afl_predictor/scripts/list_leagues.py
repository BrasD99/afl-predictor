#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import argparse
import sys
from pathlib import Path

def _bootstrap() -> None:
    path = Path(__file__).resolve().parent / "_bootstrap.py"
    spec = importlib.util.spec_from_file_location("_bootstrap", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

_bootstrap()

from afl_predictor.api.footballista import FootballistaClient, LeagueListItem
from afl_predictor.config import load_config


def _matches_filters(
    league: LeagueListItem,
    search: str | None,
    client_key: str | None,
    city: str | None,
) -> bool:
    if client_key and (not league.client or league.client.key.lower() != client_key.lower()):
        return False
    if city and (not league.city or city.lower() not in league.city.name.lower()):
        return False
    if search:
        haystack = " ".join(
            filter(
                None,
                [
                    league.name,
                    league.sports,
                    league.sports_subtype,
                    league.city.name if league.city else None,
                    league.client.name if league.client else None,
                ],
            )
        ).lower()
        if search.lower() not in haystack:
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Список лиг Footballista (для поиска ID)")
    parser.add_argument("--config", type=Path, default=None, help="Путь к config.yaml")
    parser.add_argument("--search", "-s", help="Фильтр по названию/городу (подстрока)")
    parser.add_argument("--client", "-c", default="AFL_RU", help="Фильтр по client.key (по умолчанию AFL_RU)")
    parser.add_argument("--city", help="Фильтр по городу (подстрока)")
    parser.add_argument("--all", action="store_true", help="Показать все лиги, без фильтра client")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    client = FootballistaClient(
        base_url=config.api.base_url,
        timeout_seconds=config.api.timeout_seconds,
    )

    leagues = client.get_leagues_list()
    client_key = None if args.all else args.client
    filtered = [
        league
        for league in leagues
        if _matches_filters(league, args.search, client_key, args.city)
    ]
    filtered.sort(key=lambda item: (item.city.name if item.city else "", item.name))

    if not filtered:
        print("Лиги не найдены. Попробуйте --all или другой --search.")
        return 1

    print(f"{'ID':>6}  {'Город':<20}  {'Название'}")
    print("-" * 70)
    for league in filtered:
        city_name = league.city.name if league.city else "—"
        print(f"{league.id:>6}  {city_name:<20}  {league.name}")

    print(f"\nВсего: {len(filtered)}")
    print("Скопируйте нужные id в config/config.yaml → leagues")
    return 0


if __name__ == "__main__":
    sys.exit(main())
