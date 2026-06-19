from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from afl_predictor.ml.constants import FEATURE_COLUMNS
from afl_predictor.ml.poisson import most_likely_scoreline, outcome_probabilities, result_from_score

RESULT_LABELS = {
    "H": "победа хозяев",
    "D": "ничья",
    "A": "победа гостей",
}


class ModelLoadError(RuntimeError):
    """Ошибка загрузки модели из-за несовместимости окружения."""


@dataclass(frozen=True)
class Prediction:
    home_team_id: int
    away_team_id: int
    home_team_name: str
    away_team_name: str
    predicted_home_score: int
    predicted_away_score: int
    expected_home_goals: float
    expected_away_goals: float
    predicted_result: str
    result_label: str
    probabilities: dict[str, float]


class MatchPredictor:
    def __init__(
        self,
        home_score_model,
        away_score_model,
        feature_columns: list[str] | None = None,
        league_id: int | None = None,
    ):
        self.home_score_model = home_score_model
        self.away_score_model = away_score_model
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.league_id = league_id

    @classmethod
    def load(cls, models_dir: Path) -> MatchPredictor:
        from afl_predictor.ml.env_info import format_env_mismatch_hint

        models_dir = Path(models_dir)
        for name in ("home_score_model.joblib", "away_score_model.joblib"):
            if not (models_dir / name).exists():
                raise FileNotFoundError(
                    f"Модель не найдена: {models_dir / name}. Сначала запустите train_model."
                )

        try:
            feature_columns = joblib.load(models_dir / "feature_columns.joblib")
            home_score_model = joblib.load(models_dir / "home_score_model.joblib")
            away_score_model = joblib.load(models_dir / "away_score_model.joblib")
            meta_path = models_dir / "model_meta.joblib"
            meta = joblib.load(meta_path) if meta_path.exists() else {}
        except (ValueError, ModuleNotFoundError, AttributeError) as exc:
            if "BitGenerator" in str(exc) or "numpy" in str(exc).lower():
                raise ModelLoadError(format_env_mismatch_hint(models_dir)) from exc
            raise

        return cls(
            home_score_model=home_score_model,
            away_score_model=away_score_model,
            feature_columns=feature_columns,
            league_id=meta.get("league_id"),
        )

    def predict(self, features: dict) -> Prediction:
        row = np.array([[features.get(col, 0.0) for col in self.feature_columns]], dtype=float)

        home_lambda = max(0.0, float(self.home_score_model.predict(row)[0]))
        away_lambda = max(0.0, float(self.away_score_model.predict(row)[0]))

        home_score, away_score = most_likely_scoreline(home_lambda, away_lambda)
        predicted_result = result_from_score(home_score, away_score)
        probabilities = outcome_probabilities(home_lambda, away_lambda)

        return Prediction(
            home_team_id=int(features.get("team_home_id", 0)),
            away_team_id=int(features.get("team_away_id", 0)),
            home_team_name=str(features.get("team_home_name", "")),
            away_team_name=str(features.get("team_away_name", "")),
            predicted_home_score=home_score,
            predicted_away_score=away_score,
            expected_home_goals=round(home_lambda, 2),
            expected_away_goals=round(away_lambda, 2),
            predicted_result=predicted_result,
            result_label=RESULT_LABELS.get(predicted_result, predicted_result),
            probabilities=probabilities,
        )
