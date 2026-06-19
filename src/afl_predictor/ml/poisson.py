from __future__ import annotations

import math

import numpy as np

MIN_LAMBDA = 0.05
DEFAULT_MAX_GOALS = 12
OUTCOME_CLASSES = ("H", "D", "A")


def poisson_pmf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))


def scoreline_probability(home_goals: int, away_goals: int, home_lambda: float, away_lambda: float) -> float:
    return poisson_pmf(home_goals, home_lambda) * poisson_pmf(away_goals, away_lambda)


def outcome_probabilities(
    home_lambda: float,
    away_lambda: float,
    *,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> dict[str, float]:
    """P(H), P(D), P(A) из независимого Poisson по λ хозяев и гостей."""
    home_lambda = max(MIN_LAMBDA, home_lambda)
    away_lambda = max(MIN_LAMBDA, away_lambda)

    probs = {"H": 0.0, "D": 0.0, "A": 0.0}
    total = 0.0
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            joint = scoreline_probability(home_goals, away_goals, home_lambda, away_lambda)
            total += joint
            if home_goals > away_goals:
                probs["H"] += joint
            elif home_goals < away_goals:
                probs["A"] += joint
            else:
                probs["D"] += joint

    if total <= 0:
        return {"H": 1 / 3, "D": 1 / 3, "A": 1 / 3}

    return {key: value / total for key, value in probs.items()}


def most_likely_scoreline(
    home_lambda: float,
    away_lambda: float,
    *,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> tuple[int, int]:
    """Наиболее вероятный счёт по Poisson-модели."""
    home_lambda = max(MIN_LAMBDA, home_lambda)
    away_lambda = max(MIN_LAMBDA, away_lambda)

    best_home, best_away, best_prob = 0, 0, -1.0
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            prob = scoreline_probability(home_goals, away_goals, home_lambda, away_lambda)
            if prob > best_prob:
                best_home, best_away, best_prob = home_goals, away_goals, prob

    return best_home, best_away


def result_from_score(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "H"
    if home_goals < away_goals:
        return "A"
    return "D"


def predict_outcomes_batch(
    home_lambdas: np.ndarray,
    away_lambdas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Исходы и матрица вероятностей [H, D, A] из λ регрессоров."""
    outcomes: list[str] = []
    proba_rows: list[list[float]] = []

    for home_lambda, away_lambda in zip(home_lambdas, away_lambdas, strict=True):
        home_lambda = max(0.0, float(home_lambda))
        away_lambda = max(0.0, float(away_lambda))
        home_score, away_score = most_likely_scoreline(home_lambda, away_lambda)
        outcomes.append(result_from_score(home_score, away_score))
        probs = outcome_probabilities(home_lambda, away_lambda)
        proba_rows.append([probs[label] for label in OUTCOME_CLASSES])

    return np.array(outcomes), np.array(proba_rows)
