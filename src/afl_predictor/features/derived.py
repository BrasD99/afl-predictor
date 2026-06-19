from __future__ import annotations


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def unbeaten_streak(history: list[tuple[int, int, int]]) -> int:
    streak = 0
    for gf, ga, _ in reversed(history):
        if gf >= ga:
            streak += 1
        else:
            break
    return streak


def h2h_stats(
    recent_h2h: list[tuple[int, int, int, int]],
    current_home_id: int,
) -> dict[str, float]:
    if not recent_h2h:
        return {
            "h2h_matches": 0.0,
            "h2h_home_points": 0.0,
            "h2h_avg_total_goals": 0.0,
            "h2h_home_win_rate": 0.0,
        }

    home_points = 0
    home_wins = 0
    total_goals = 0
    for past_home_id, past_away_id, past_home_score, past_away_score in recent_h2h:
        total_goals += past_home_score + past_away_score
        if past_home_id == current_home_id:
            if past_home_score > past_away_score:
                home_wins += 1
                home_points += 3
            elif past_home_score == past_away_score:
                home_points += 1
        else:
            if past_away_score > past_home_score:
                home_wins += 1
                home_points += 3
            elif past_away_score == past_home_score:
                home_points += 1

    count = len(recent_h2h)
    return {
        "h2h_matches": float(count),
        "h2h_home_points": float(home_points),
        "h2h_avg_total_goals": total_goals / count,
        "h2h_home_win_rate": home_wins / count,
    }


def enrich_features(base: dict[str, float]) -> dict[str, float]:
    """Производные признаки: диффы, Poisson-ожидание голов, флаги усталости."""
    home_gf_pg = _safe_div(base["home_season_goals_for"], base["home_season_played"])
    away_gf_pg = _safe_div(base["away_season_goals_for"], base["away_season_played"])
    home_ga_pg = _safe_div(base["home_season_goals_against"], base["home_season_played"])
    away_ga_pg = _safe_div(base["away_season_goals_against"], base["away_season_played"])

    home_home_gf_pg = _safe_div(base["home_home_goals_for"], base["home_home_played"])
    home_home_ga_pg = _safe_div(base["home_home_goals_against"], base["home_home_played"])
    away_away_gf_pg = _safe_div(base["away_away_goals_for"], base["away_away_played"])
    away_away_ga_pg = _safe_div(base["away_away_goals_against"], base["away_away_played"])

    home_home_ppg = _safe_div(base["home_home_points"], base["home_home_played"])
    away_away_ppg = _safe_div(base["away_away_points"], base["away_away_played"])

    # Классическая эвристика ожидаемых голов (атака × оборона соперника)
    exp_home_goals = (home_gf_pg + away_ga_pg) / 2.0
    exp_away_goals = (away_gf_pg + home_ga_pg) / 2.0
    exp_home_venue = (home_home_gf_pg + away_away_ga_pg) / 2.0
    exp_away_venue = (away_away_gf_pg + home_home_ga_pg) / 2.0

    home_rest = base["home_rest_days"]
    away_rest = base["away_rest_days"]

    return {
        "home_season_gf_pg": home_gf_pg,
        "away_season_gf_pg": away_gf_pg,
        "home_season_ga_pg": home_ga_pg,
        "away_season_ga_pg": away_ga_pg,
        "home_home_ppg": home_home_ppg,
        "away_away_ppg": away_away_ppg,
        "home_home_gf_pg": home_home_gf_pg,
        "home_home_ga_pg": home_home_ga_pg,
        "away_away_gf_pg": away_away_gf_pg,
        "away_away_ga_pg": away_away_ga_pg,
        "season_ppg_diff": base["home_season_ppg"] - base["away_season_ppg"],
        "season_gd_diff": base["home_season_goal_diff"] - base["away_season_goal_diff"],
        "form_points_diff": base["home_form_points"] - base["away_form_points"],
        "form_gd_diff": base["home_form_goal_diff"] - base["away_form_goal_diff"],
        "attack_strength_diff": home_gf_pg - away_gf_pg,
        "defense_strength_diff": home_ga_pg - away_ga_pg,
        "venue_ppg_diff": home_home_ppg - away_away_ppg,
        "venue_form_points_diff": base["home_venue_form_points"] - base["away_venue_form_points"],
        "exp_home_goals": exp_home_goals,
        "exp_away_goals": exp_away_goals,
        "exp_total_goals": exp_home_goals + exp_away_goals,
        "exp_home_venue_goals": exp_home_venue,
        "exp_away_venue_goals": exp_away_venue,
        "home_short_rest": 1.0 if home_rest < 5.0 else 0.0,
        "away_short_rest": 1.0 if away_rest < 5.0 else 0.0,
        "home_unbeaten_streak": base["home_unbeaten_streak"],
        "away_unbeaten_streak": base["away_unbeaten_streak"],
    }
