from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from afl_predictor.db.models import Match
from afl_predictor.features.derived import enrich_features, h2h_stats, unbeaten_streak
from afl_predictor.features.ratings import RatingsTracker

POINTS_WIN = 3
POINTS_DRAW = 1
POINTS_LOSS = 0


@dataclass(frozen=True)
class TeamSnapshot:
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0
    home_played: int = 0
    home_points: int = 0
    home_goals_for: int = 0
    home_goals_against: int = 0
    away_played: int = 0
    away_points: int = 0
    away_goals_for: int = 0
    away_goals_against: int = 0
    last_match_at: datetime | None = None


def _as_naive(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _days_between(later: datetime, earlier: datetime) -> float:
    return (_as_naive(later) - _as_naive(earlier)).total_seconds() / 86400


def _match_points(home_score: int, away_score: int, for_home: bool) -> int:
    if home_score == away_score:
        return POINTS_DRAW
    home_won = home_score > away_score
    if for_home:
        return POINTS_WIN if home_won else POINTS_LOSS
    return POINTS_WIN if not home_won else POINTS_LOSS


def _match_result(home_score: int, away_score: int) -> str:
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"


def _rolling_form(history: list[tuple[int, int, int]], window: int) -> dict[str, float]:
    recent = history[-window:]
    if not recent:
        return {
            "points": 0.0,
            "goals_for": 0.0,
            "goals_against": 0.0,
            "goal_diff": 0.0,
            "wins": 0.0,
        }

    goals_for = sum(item[0] for item in recent)
    goals_against = sum(item[1] for item in recent)
    points = sum(item[2] for item in recent)
    wins = sum(1 for gf, ga, _ in recent if gf > ga)
    count = len(recent)

    return {
        "points": points / count,
        "goals_for": goals_for / count,
        "goals_against": goals_against / count,
        "goal_diff": (goals_for - goals_against) / count,
        "wins": wins / count,
    }


def _compose_features(
    *,
    home_stats: TeamSnapshot,
    away_stats: TeamSnapshot,
    home_form: dict[str, float],
    away_form: dict[str, float],
    home_venue_form: dict[str, float],
    away_venue_form: dict[str, float],
    rating_features: dict[str, float],
    home_rest_days: float | None,
    away_rest_days: float | None,
    recent_h2h: list[tuple[int, int, int, int]],
    current_home_id: int,
    home_history: list[tuple[int, int, int]],
    away_history: list[tuple[int, int, int]],
) -> dict[str, float]:
    home_rest = home_rest_days if home_rest_days is not None else 14.0
    away_rest = away_rest_days if away_rest_days is not None else 14.0

    base: dict[str, float] = {
        "home_season_points": float(home_stats.points),
        "away_season_points": float(away_stats.points),
        "home_season_goal_diff": float(home_stats.goals_for - home_stats.goals_against),
        "away_season_goal_diff": float(away_stats.goals_for - away_stats.goals_against),
        "home_season_played": float(home_stats.played),
        "away_season_played": float(away_stats.played),
        "home_season_ppg": home_stats.points / home_stats.played if home_stats.played else 0.0,
        "away_season_ppg": away_stats.points / away_stats.played if away_stats.played else 0.0,
        "home_season_goals_for": float(home_stats.goals_for),
        "away_season_goals_for": float(away_stats.goals_for),
        "home_season_goals_against": float(home_stats.goals_against),
        "away_season_goals_against": float(away_stats.goals_against),
        "home_home_played": float(home_stats.home_played),
        "away_away_played": float(away_stats.away_played),
        "home_home_points": float(home_stats.home_points),
        "away_away_points": float(away_stats.away_points),
        "home_home_goals_for": float(home_stats.home_goals_for),
        "home_home_goals_against": float(home_stats.home_goals_against),
        "away_away_goals_for": float(away_stats.away_goals_for),
        "away_away_goals_against": float(away_stats.away_goals_against),
        "home_form_points": home_form["points"],
        "away_form_points": away_form["points"],
        "home_form_goals_for": home_form["goals_for"],
        "away_form_goals_for": away_form["goals_for"],
        "home_form_goals_against": home_form["goals_against"],
        "away_form_goals_against": away_form["goals_against"],
        "home_form_goal_diff": home_form["goal_diff"],
        "away_form_goal_diff": away_form["goal_diff"],
        "home_form_wins": home_form["wins"],
        "away_form_wins": away_form["wins"],
        "home_venue_form_points": home_venue_form["points"],
        "away_venue_form_points": away_venue_form["points"],
        "home_venue_form_gd": home_venue_form["goal_diff"],
        "away_venue_form_gd": away_venue_form["goal_diff"],
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "rest_days_diff": home_rest - away_rest,
        "home_unbeaten_streak": float(unbeaten_streak(home_history)),
        "away_unbeaten_streak": float(unbeaten_streak(away_history)),
        **h2h_stats(recent_h2h, current_home_id),
        **rating_features,
    }
    return {**base, **enrich_features(base)}


class FeatureBuilder:
    def __init__(
        self,
        season_championship_map: dict[int, int],
        form_window: int = 5,
        h2h_window: int = 5,
    ):
        self.season_championship_map = season_championship_map
        self.form_window = form_window
        self.h2h_window = h2h_window

    def build(self, matches: list[Match]):
        import pandas as pd

        rows: list[dict] = []
        season_state: dict[int, dict[int, TeamSnapshot]] = {}
        season_form: dict[int, dict[int, list[tuple[int, int, int]]]] = {}
        season_home_form: dict[int, dict[int, list[tuple[int, int, int]]]] = {}
        season_away_form: dict[int, dict[int, list[tuple[int, int, int]]]] = {}
        season_h2h: dict[int, dict[tuple[int, int], list[tuple[int, int, int, int]]]] = {}
        ratings = RatingsTracker(self.season_championship_map, self.form_window)

        for match in matches:
            season_id = match.season_id
            home_id = match.team_home_id
            away_id = match.team_away_id

            season_state.setdefault(season_id, {})
            season_form.setdefault(season_id, {})
            season_home_form.setdefault(season_id, {})
            season_away_form.setdefault(season_id, {})
            season_h2h.setdefault(season_id, {})

            home_stats = season_state[season_id].setdefault(home_id, TeamSnapshot())
            away_stats = season_state[season_id].setdefault(away_id, TeamSnapshot())
            home_history = season_form[season_id].setdefault(home_id, [])
            away_history = season_form[season_id].setdefault(away_id, [])
            home_at_home_history = season_home_form[season_id].setdefault(home_id, [])
            away_on_road_history = season_away_form[season_id].setdefault(away_id, [])
            pair_key = tuple(sorted((home_id, away_id)))
            h2h_history = season_h2h[season_id].setdefault(pair_key, [])

            home_form = _rolling_form(home_history, self.form_window)
            away_form = _rolling_form(away_history, self.form_window)
            home_venue_form = _rolling_form(home_at_home_history, self.form_window)
            away_venue_form = _rolling_form(away_on_road_history, self.form_window)
            rating_features = ratings.snapshot(home_id, away_id, season_id, season_id)

            home_rest_days = (
                _days_between(match.played_at, home_stats.last_match_at)
                if home_stats.last_match_at
                else None
            )
            away_rest_days = (
                _days_between(match.played_at, away_stats.last_match_at)
                if away_stats.last_match_at
                else None
            )

            recent_h2h = h2h_history[-self.h2h_window :]
            feature_values = _compose_features(
                home_stats=home_stats,
                away_stats=away_stats,
                home_form=home_form,
                away_form=away_form,
                home_venue_form=home_venue_form,
                away_venue_form=away_venue_form,
                rating_features=rating_features,
                home_rest_days=home_rest_days,
                away_rest_days=away_rest_days,
                recent_h2h=recent_h2h,
                current_home_id=home_id,
                home_history=home_history,
                away_history=away_history,
            )

            rows.append(
                {
                    "match_id": match.id,
                    "external_id": match.external_id,
                    "season_id": season_id,
                    "played_at": match.played_at,
                    "team_home_id": home_id,
                    "team_away_id": away_id,
                    "team_home_name": match.team_home.name,
                    "team_away_name": match.team_away.name,
                    **feature_values,
                    "score_home": match.score_ft_home,
                    "score_away": match.score_ft_away,
                    "result": _match_result(match.score_ft_home, match.score_ft_away),
                }
            )

            self._apply_season_match(
                match,
                season_state[season_id],
                season_form[season_id],
                season_home_form[season_id],
                season_away_form[season_id],
                h2h_history,
            )
            ratings.observe(match)

        return pd.DataFrame(rows)

    def build_for_teams(
        self,
        all_matches: list[Match],
        home_id: int,
        away_id: int,
        *,
        season_id: int | None = None,
        played_at: datetime | None = None,
        home_name: str = "",
        away_name: str = "",
    ) -> dict:
        """Признаки для гипотетического матча. Elo — по всей истории, сезон — по season_id."""
        if played_at is None:
            played_at = datetime.now(timezone.utc).replace(tzinfo=None)

        ratings = RatingsTracker(self.season_championship_map, self.form_window)
        team_last_season: dict[int, int] = {}

        season_state: dict[int, TeamSnapshot] = {}
        season_form: dict[int, list[tuple[int, int, int]]] = {}
        season_home_form: dict[int, list[tuple[int, int, int]]] = {}
        season_away_form: dict[int, list[tuple[int, int, int]]] = {}
        h2h_history: list[tuple[int, int, int, int]] = []

        for match in all_matches:
            ratings.observe(match)
            team_last_season[match.team_home_id] = match.season_id
            team_last_season[match.team_away_id] = match.season_id

            if season_id is None or match.season_id == season_id:
                self._apply_season_match(
                    match,
                    season_state,
                    season_form,
                    season_home_form,
                    season_away_form,
                    h2h_history,
                )

        home_stats = season_state.get(home_id, TeamSnapshot())
        away_stats = season_state.get(away_id, TeamSnapshot())
        home_history = season_form.get(home_id, [])
        away_history = season_form.get(away_id, [])
        home_at_home_history = season_home_form.get(home_id, [])
        away_on_road_history = season_away_form.get(away_id, [])

        home_form = _rolling_form(home_history, self.form_window)
        away_form = _rolling_form(away_history, self.form_window)
        home_venue_form = _rolling_form(home_at_home_history, self.form_window)
        away_venue_form = _rolling_form(away_on_road_history, self.form_window)

        home_last_season_id = team_last_season.get(home_id, season_id or 0)
        away_last_season_id = team_last_season.get(away_id, season_id or 0)
        rating_features = ratings.snapshot(
            home_id,
            away_id,
            home_last_season_id,
            away_last_season_id,
        )

        home_rest_days = (
            _days_between(played_at, home_stats.last_match_at)
            if home_stats.last_match_at
            else None
        )
        away_rest_days = (
            _days_between(played_at, away_stats.last_match_at)
            if away_stats.last_match_at
            else None
        )

        recent_h2h = h2h_history[-self.h2h_window :]
        feature_values = _compose_features(
            home_stats=home_stats,
            away_stats=away_stats,
            home_form=home_form,
            away_form=away_form,
            home_venue_form=home_venue_form,
            away_venue_form=away_venue_form,
            rating_features=rating_features,
            home_rest_days=home_rest_days,
            away_rest_days=away_rest_days,
            recent_h2h=recent_h2h,
            current_home_id=home_id,
            home_history=home_history,
            away_history=away_history,
        )

        return {
            "team_home_id": home_id,
            "team_away_id": away_id,
            "team_home_name": home_name,
            "team_away_name": away_name,
            **feature_values,
        }

    def _apply_season_match(
        self,
        match: Match,
        season_state: dict[int, TeamSnapshot],
        season_form: dict[int, list[tuple[int, int, int]]],
        season_home_form: dict[int, list[tuple[int, int, int]]],
        season_away_form: dict[int, list[tuple[int, int, int]]],
        h2h_history: list[tuple[int, int, int, int]],
    ) -> None:
        home_id = match.team_home_id
        away_id = match.team_away_id
        home_score = match.score_ft_home
        away_score = match.score_ft_away

        home_stats = season_state.get(home_id, TeamSnapshot())
        away_stats = season_state.get(away_id, TeamSnapshot())

        season_state[home_id] = TeamSnapshot(
            played=home_stats.played + 1,
            wins=home_stats.wins + (1 if home_score > away_score else 0),
            draws=home_stats.draws + (1 if home_score == away_score else 0),
            losses=home_stats.losses + (1 if home_score < away_score else 0),
            goals_for=home_stats.goals_for + home_score,
            goals_against=home_stats.goals_against + away_score,
            points=home_stats.points + _match_points(home_score, away_score, True),
            home_played=home_stats.home_played + 1,
            home_points=home_stats.home_points + _match_points(home_score, away_score, True),
            home_goals_for=home_stats.home_goals_for + home_score,
            home_goals_against=home_stats.home_goals_against + away_score,
            away_played=home_stats.away_played,
            away_points=home_stats.away_points,
            away_goals_for=home_stats.away_goals_for,
            away_goals_against=home_stats.away_goals_against,
            last_match_at=match.played_at,
        )
        season_state[away_id] = TeamSnapshot(
            played=away_stats.played + 1,
            wins=away_stats.wins + (1 if away_score > home_score else 0),
            draws=away_stats.draws + (1 if away_score == home_score else 0),
            losses=away_stats.losses + (1 if away_score < home_score else 0),
            goals_for=away_stats.goals_for + away_score,
            goals_against=away_stats.goals_against + home_score,
            points=away_stats.points + _match_points(home_score, away_score, False),
            home_played=away_stats.home_played,
            home_points=away_stats.home_points,
            home_goals_for=away_stats.home_goals_for,
            home_goals_against=away_stats.home_goals_against,
            away_played=away_stats.away_played + 1,
            away_points=away_stats.away_points + _match_points(home_score, away_score, False),
            away_goals_for=away_stats.away_goals_for + away_score,
            away_goals_against=away_stats.away_goals_against + home_score,
            last_match_at=match.played_at,
        )

        home_pts = _match_points(home_score, away_score, True)
        away_pts = _match_points(home_score, away_score, False)

        season_form.setdefault(home_id, []).append((home_score, away_score, home_pts))
        season_form.setdefault(away_id, []).append((away_score, home_score, away_pts))
        season_home_form.setdefault(home_id, []).append((home_score, away_score, home_pts))
        season_away_form.setdefault(away_id, []).append((away_score, home_score, away_pts))
        h2h_history.append((home_id, away_id, home_score, away_score))


def build_inference_features(
    all_matches: list[Match],
    home_id: int,
    away_id: int,
    *,
    season_championship_map: dict[int, int],
    season_id: int | None = None,
    form_window: int = 5,
    h2h_window: int = 5,
    played_at: datetime | None = None,
    home_name: str = "",
    away_name: str = "",
) -> dict:
    builder = FeatureBuilder(
        season_championship_map=season_championship_map,
        form_window=form_window,
        h2h_window=h2h_window,
    )
    return builder.build_for_teams(
        all_matches,
        home_id,
        away_id,
        season_id=season_id,
        played_at=played_at,
        home_name=home_name,
        away_name=away_name,
    )


def build_training_dataset(
    matches: list[Match],
    season_championship_map: dict[int, int],
    form_window: int = 5,
    h2h_window: int = 5,
):
    builder = FeatureBuilder(
        season_championship_map=season_championship_map,
        form_window=form_window,
        h2h_window=h2h_window,
    )
    return builder.build(matches)
