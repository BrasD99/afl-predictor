#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import argparse
import csv
import sys
from pathlib import Path


def _bootstrap() -> None:
    path = Path(__file__).resolve().parent / "_bootstrap.py"
    spec = importlib.util.spec_from_file_location("_bootstrap", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


_bootstrap()

from sqlalchemy import text

from afl_predictor.config import load_config
from afl_predictor.db import get_session_factory, init_db

TEAMS_QUERY = """
WITH team_matches AS (
    SELECT
        team_home_id AS team_id,
        score_ft_home AS goals_for,
        score_ft_away AS goals_against,
        season_id,
        played_at,
        CASE WHEN score_ft_home > score_ft_away THEN 1 ELSE 0 END AS win,
        CASE WHEN score_ft_home = score_ft_away THEN 1 ELSE 0 END AS draw
    FROM matches
    WHERE score_ft_home IS NOT NULL AND score_ft_away IS NOT NULL
    UNION ALL
    SELECT
        team_away_id,
        score_ft_away,
        score_ft_home,
        season_id,
        played_at,
        CASE WHEN score_ft_away > score_ft_home THEN 1 ELSE 0 END,
        CASE WHEN score_ft_away = score_ft_home THEN 1 ELSE 0 END
    FROM matches
    WHERE score_ft_home IS NOT NULL AND score_ft_away IS NOT NULL
),
team_leagues AS (
    SELECT DISTINCT
        tm.team_id,
        l.id AS league_id,
        l.name AS league_name
    FROM team_matches tm
    JOIN seasons s ON s.id = tm.season_id
    JOIN championships c ON c.id = s.championship_id
    JOIN leagues l ON l.id = c.league_id
)
SELECT
    t.id,
    t.name,
    t.short_name,
    t.logo_id,
    COUNT(tm.team_id) AS matches_played,
    COALESCE(SUM(tm.win), 0) AS wins,
    COALESCE(SUM(tm.draw), 0) AS draws,
    COALESCE(SUM(tm.goals_for), 0) AS goals_for,
    COALESCE(SUM(tm.goals_against), 0) AS goals_against,
    MAX(tm.played_at) AS last_match_at,
    GROUP_CONCAT(DISTINCT tl.league_name) AS leagues,
    GROUP_CONCAT(DISTINCT CAST(tl.league_id AS TEXT)) AS league_ids
FROM teams t
LEFT JOIN team_matches tm ON tm.team_id = t.id
LEFT JOIN team_leagues tl ON tl.team_id = t.id
WHERE (:search = '' OR LOWER(t.name) LIKE '%' || LOWER(:search) || '%'
       OR LOWER(COALESCE(t.short_name, '')) LIKE '%' || LOWER(:search) || '%')
  AND (:league_id = 0 OR tl.league_id = :league_id)
GROUP BY t.id, t.name, t.short_name, t.logo_id
HAVING (:league_id = 0 OR COUNT(tm.team_id) > 0)
ORDER BY matches_played DESC, t.name
"""

CSV_COLUMNS = [
    "id",
    "name",
    "short_name",
    "logo_id",
    "matches_played",
    "wins",
    "draws",
    "losses",
    "goals_for",
    "goals_against",
    "goal_diff",
    "last_match_at",
    "leagues",
    "league_ids",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Экспорт команд и их Footballista ID в CSV",
    )
    parser.add_argument("--config", type=Path, default=None, help="Путь к config.yaml")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Путь к CSV (по умолчанию data/teams.csv)",
    )
    parser.add_argument("--search", "-s", default="", help="Фильтр по названию команды")
    parser.add_argument("--league-id", type=int, default=None, help="Только команды из лиги (по умолчанию — из config)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    league = config.league
    db_path = config.database.path
    output_path = args.output or (config.database.path.parent / "teams.csv")

    if not db_path.exists():
        print(f"База данных не найдена: {db_path}", file=sys.stderr)
        return 1

    init_db(str(db_path))
    session_factory = get_session_factory(str(db_path))

    league_id = args.league_id if args.league_id is not None else league.id

    with session_factory() as session:
        rows = session.execute(
            text(TEAMS_QUERY),
            {"search": args.search, "league_id": league_id},
        ).mappings().all()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            matches = int(row["matches_played"] or 0)
            wins = int(row["wins"] or 0)
            draws = int(row["draws"] or 0)
            goals_for = int(row["goals_for"] or 0)
            goals_against = int(row["goals_against"] or 0)
            writer.writerow(
                {
                    "id": row["id"],
                    "name": row["name"],
                    "short_name": row["short_name"] or "",
                    "logo_id": row["logo_id"] or "",
                    "matches_played": matches,
                    "wins": wins,
                    "draws": draws,
                    "losses": matches - wins - draws,
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "goal_diff": goals_for - goals_against,
                    "last_match_at": row["last_match_at"] or "",
                    "leagues": row["leagues"] or "",
                    "league_ids": row["league_ids"] or "",
                }
            )

    print(f"Сохранено {len(rows)} команд → {output_path}")
    print(f"Лига: {league.name or league.id} (id={league_id})")
    print("id — внутренний Footballista ID (используйте в predict_match)")
    if rows:
        print("\nТоп-5 по количеству матчей:")
        for row in rows[:5]:
            print(f"  {row['id']:>6}  {row['name']}  ({row['matches_played']} матчей)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
