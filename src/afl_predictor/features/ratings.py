from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from afl_predictor.db.models import Match

BASE_ELO = 1500.0
HOME_ADVANTAGE_ELO = 65.0
K_FACTOR = 32.0


@dataclass(frozen=True)
class FormRecord:
    goals_for: int
    goals_against: int
    points: int
    opponent_elo: float
    elo_performance: float


def _expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def _actual_scores(home_score: int, away_score: int) -> tuple[float, float]:
    if home_score > away_score:
        return 1.0, 0.0
    if home_score < away_score:
        return 0.0, 1.0
    return 0.5, 0.5


def _match_points(home_score: int, away_score: int, for_home: bool) -> int:
    if home_score == away_score:
        return 1
    home_won = home_score > away_score
    if for_home:
        return 3 if home_won else 0
    return 3 if not home_won else 0


class RatingsTracker:
    """Глобальный Elo, сила чемпионатов и форма с учётом силы соперника."""

    def __init__(
        self,
        season_championship_map: dict[int, int],
        form_window: int = 5,
    ):
        self.season_championship_map = season_championship_map
        self.form_window = form_window
        self.team_elo: dict[int, float] = {}
        self.championship_teams: dict[int, set[int]] = defaultdict(set)
        self.championship_strength: dict[int, float] = {}
        self.team_form: dict[int, list[FormRecord]] = defaultdict(list)

    def _elo(self, team_id: int) -> float:
        return self.team_elo.get(team_id, BASE_ELO)

    def _championship_id(self, season_id: int) -> int:
        return self.season_championship_map.get(season_id, 0)

    def _champ_strength(self, championship_id: int) -> float:
        if championship_id == 0:
            return 1.0
        return self.championship_strength.get(championship_id, 1.0)

    def _recompute_championship_strength(self, championship_id: int) -> None:
        teams = self.championship_teams.get(championship_id)
        if not teams:
            return
        avg_elo = sum(self._elo(team_id) for team_id in teams) / len(teams)
        self.championship_strength[championship_id] = avg_elo / BASE_ELO

    def _rolling_form(self, team_id: int) -> dict[str, float]:
        recent = self.team_form[team_id][-self.form_window :]
        if not recent:
            return {
                "opp_adj_points": 0.0,
                "elo_performance": 0.0,
                "opp_adj_goal_diff": 0.0,
            }

        count = len(recent)
        opp_adj_points = sum(r.points * (r.opponent_elo / BASE_ELO) for r in recent) / count
        elo_performance = sum(r.elo_performance for r in recent) / count
        opp_adj_goal_diff = sum(
            (r.goals_for - r.goals_against) * (r.opponent_elo / BASE_ELO) for r in recent
        ) / count
        return {
            "opp_adj_points": opp_adj_points,
            "elo_performance": elo_performance,
            "opp_adj_goal_diff": opp_adj_goal_diff,
        }

    def snapshot(
        self,
        home_id: int,
        away_id: int,
        home_season_id: int,
        away_season_id: int,
    ) -> dict[str, float]:
        home_elo = self._elo(home_id)
        away_elo = self._elo(away_id)

        home_champ_id = self._championship_id(home_season_id)
        away_champ_id = self._championship_id(away_season_id)
        home_champ = self._champ_strength(home_champ_id)
        away_champ = self._champ_strength(away_champ_id)

        home_adj = home_elo * home_champ
        away_adj = away_elo * away_champ
        home_form = self._rolling_form(home_id)
        away_form = self._rolling_form(away_id)
        home_elo_win_prob = _expected_score(home_elo + HOME_ADVANTAGE_ELO, away_elo)

        return {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "elo_diff": home_elo - away_elo + HOME_ADVANTAGE_ELO,
            "home_elo_win_prob": home_elo_win_prob,
            "home_elo_adj": home_adj,
            "away_elo_adj": away_adj,
            "elo_adj_diff": home_adj - away_adj + HOME_ADVANTAGE_ELO,
            "home_champ_strength": home_champ,
            "away_champ_strength": away_champ,
            "champ_strength_diff": home_champ - away_champ,
            "home_form_opp_adj_points": home_form["opp_adj_points"],
            "away_form_opp_adj_points": away_form["opp_adj_points"],
            "home_form_elo_perf": home_form["elo_performance"],
            "away_form_elo_perf": away_form["elo_performance"],
            "home_form_opp_adj_gd": home_form["opp_adj_goal_diff"],
            "away_form_opp_adj_gd": away_form["opp_adj_goal_diff"],
        }

    def observe(self, match: Match) -> None:
        home_id = match.team_home_id
        away_id = match.team_away_id
        home_score = match.score_ft_home
        away_score = match.score_ft_away

        home_elo = self._elo(home_id)
        away_elo = self._elo(away_id)

        expected_home = _expected_score(home_elo + HOME_ADVANTAGE_ELO, away_elo)
        actual_home, actual_away = _actual_scores(home_score, away_score)

        home_perf = actual_home - expected_home
        away_perf = actual_away - (1.0 - expected_home)

        self.team_elo[home_id] = home_elo + K_FACTOR * home_perf
        self.team_elo[away_id] = away_elo + K_FACTOR * away_perf

        home_champ = self._championship_id(match.season_id)
        if home_champ:
            self.championship_teams[home_champ].add(home_id)
            self.championship_teams[home_champ].add(away_id)
            self._recompute_championship_strength(home_champ)

        self.team_form[home_id].append(
            FormRecord(
                goals_for=home_score,
                goals_against=away_score,
                points=_match_points(home_score, away_score, True),
                opponent_elo=away_elo,
                elo_performance=home_perf,
            )
        )
        self.team_form[away_id].append(
            FormRecord(
                goals_for=away_score,
                goals_against=home_score,
                points=_match_points(home_score, away_score, False),
                opponent_elo=home_elo,
                elo_performance=away_perf,
            )
        )
