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

from afl_predictor.config import load_config
from afl_predictor.db import MatchRepository, get_session_factory, init_db
from afl_predictor.features.builder import build_inference_features
from afl_predictor.ml.predictor import MatchPredictor, ModelLoadError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Предсказание исхода матча по ID команд (хозяева vs гости)",
    )
    parser.add_argument("home_id", type=int, help="ID команды-хозяина")
    parser.add_argument("away_id", type=int, help="ID команды-гостя")
    parser.add_argument("--config", type=Path, default=None, help="Путь к config.yaml")
    parser.add_argument(
        "--season-id",
        type=int,
        default=None,
        help="ID сезона для расчёта формы (по умолчанию — последний общий сезон)",
    )
    args = parser.parse_args(argv)

    if args.home_id == args.away_id:
        print("Ошибка: home_id и away_id должны различаться", file=sys.stderr)
        return 1

    config = load_config(args.config)
    league = config.league
    db_path = config.database.path

    if not db_path.exists():
        print(f"База данных не найдена: {db_path}. Сначала запустите fetch_matches.", file=sys.stderr)
        return 1

    models_dir = config.training.models_dir
    try:
        predictor = MatchPredictor.load(models_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ModelLoadError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if predictor.league_id is not None and predictor.league_id != league.id:
        print(
            f"Предупреждение: модель обучена на лиге {predictor.league_id}, "
            f"в конфиге лига {league.id}",
            file=sys.stderr,
        )

    init_db(str(db_path))
    session_factory = get_session_factory(str(db_path))

    with session_factory() as session:
        repo = MatchRepository(session)

        home_team = repo.get_team(args.home_id)
        away_team = repo.get_team(args.away_id)
        if home_team is None:
            print(f"Команда home_id={args.home_id} не найдена в БД", file=sys.stderr)
            return 1
        if away_team is None:
            print(f"Команда away_id={args.away_id} не найдена в БД", file=sys.stderr)
            return 1

        season_id = args.season_id
        if season_id is None:
            season_id = repo.find_latest_common_season(
                args.home_id,
                args.away_id,
                config.fetch.finished_state_code,
                league_id=league.id,
            )

        all_matches = repo.list_finished_matches(
            config.fetch.finished_state_code,
            league_id=league.id,
        )
        season_championship_map = repo.get_season_championship_map()

        if not all_matches:
            print("Нет завершённых матчей в БД", file=sys.stderr)
            return 1

        features = build_inference_features(
            all_matches,
            args.home_id,
            args.away_id,
            season_championship_map=season_championship_map,
            season_id=season_id,
            form_window=config.training.form_window,
            h2h_window=config.training.h2h_window,
            home_name=home_team.name,
            away_name=away_team.name,
        )

        prediction = predictor.predict(features)

    season_note = f"сезон {season_id}" if season_id else "все сезоны"
    print(f"Лига: {league.name or league.id} (id={league.id})")
    print(f"Матч: {prediction.home_team_name} (id={prediction.home_team_id}) vs "
          f"{prediction.away_team_name} (id={prediction.away_team_id})")
    print(f"Контекст: {season_note}, матчей в истории: {len(all_matches)}")
    print(f"Elo: {features['home_elo']:.0f} (чемп.×{features['home_champ_strength']:.2f}) vs "
          f"{features['away_elo']:.0f} (чемп.×{features['away_champ_strength']:.2f}), "
          f"adj diff: {features['elo_adj_diff']:+.0f}")
    print()
    print(
        f"Прогноз счёта: {prediction.predicted_home_score} : {prediction.predicted_away_score} "
        f"(ожидание: {prediction.expected_home_goals:.1f} : {prediction.expected_away_goals:.1f})"
    )
    print(f"Прогноз исхода:  {prediction.result_label} ({prediction.predicted_result})")

    if prediction.probabilities:
        print()
        print("Вероятности:")
        for code in ("H", "D", "A"):
            if code in prediction.probabilities:
                label = {"H": "хозяева", "D": "ничья", "A": "гости"}[code]
                print(f"  {label}: {prediction.probabilities[code] * 100:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
