from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

from afl_predictor.ml.constants import FEATURE_COLUMNS
from afl_predictor.ml.env_info import save_env_info
from afl_predictor.ml.poisson import OUTCOME_CLASSES, predict_outcomes_batch
from afl_predictor.ml.reports import EvaluationBundle, generate_training_reports

logger = logging.getLogger(__name__)

MODEL_TYPE = "poisson_score"

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

DEFAULT_POISSON_PARAMS: dict[str, Any] = {
    **DEFAULT_LGBM_PARAMS,
    "objective": "poisson",
}


@dataclass
class TrainingReport:
    outcome_accuracy: float
    home_score_mae: float
    away_score_mae: float
    train_size: int
    test_size: int
    models_dir: Path
    league_id: int
    league_name: str | None
    split_type: str
    model_type: str = MODEL_TYPE
    best_params: dict[str, Any] = field(default_factory=dict)
    report_paths: list[Path] = field(default_factory=list)
    reports_dir: Path | None = None


def temporal_split(
    dataset: pd.DataFrame,
    test_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = dataset.sort_values("played_at").reset_index(drop=True)
    if len(ordered) < 20:
        return ordered, ordered.iloc[0:0]

    split_at = max(1, int(len(ordered) * (1.0 - test_fraction)))
    if split_at >= len(ordered):
        split_at = len(ordered) - 1
    return ordered.iloc[:split_at], ordered.iloc[split_at:]


def _optuna_progress_callback(label: str, n_trials: int) -> Callable:
    def callback(study, trial) -> None:
        if trial.value is None:
            return
        logger.info(
            "  Optuna [%s] trial %s/%s — MAE=%.3f, лучший=%.3f",
            label,
            trial.number + 1,
            n_trials,
            trial.value,
            study.best_value,
        )

    return callback


def _lgb_progress_callback(label: str, period: int = 50) -> Callable:
    def callback(env) -> None:
        iteration = env.iteration + 1
        total = env.end_iteration
        if iteration % period == 0 or iteration == total:
            logger.info("  %s: дерево %s / %s", label, iteration, total)

    callback.order = 10  # type: ignore[attr-defined]
    return callback


def _fit_regressor(
    model: LGBMRegressor,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    label: str,
    log_every: int = 50,
) -> LGBMRegressor:
    n_trees = model.get_params().get("n_estimators", "?")
    logger.info("Обучение: %s (%s деревьев)...", label, n_trees)
    started = time.perf_counter()
    model.fit(
        features,
        target,
        callbacks=[_lgb_progress_callback(label, period=log_every)],
    )
    elapsed = time.perf_counter() - started
    logger.info("Готово: %s за %.1f с", label, elapsed)
    return model


def _tune_poisson_regressor(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    label: str,
    random_state: int,
    n_trials: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info("Optuna [%s]: %s trials, timeout %s с", label, n_trials, timeout_seconds)

    inner_split = max(1, int(len(x_train) * 0.85))
    x_fit, x_val = x_train.iloc[:inner_split], x_train.iloc[inner_split:]
    y_fit, y_val = y_train.iloc[:inner_split], y_train.iloc[inner_split:]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "poisson",
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 64),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": random_state,
            "verbose": -1,
        }
        model = LGBMRegressor(**params)
        model.fit(x_fit, y_fit)
        pred = model.predict(x_val)
        return float(mean_absolute_error(y_val, pred))

    study = optuna.create_study(direction="minimize")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_seconds,
        callbacks=[_optuna_progress_callback(label, n_trials)],
        show_progress_bar=False,
    )
    logger.info("Optuna [%s]: лучший MAE=%.3f", label, study.best_value)
    return {
        **DEFAULT_POISSON_PARAMS,
        **study.best_params,
        "objective": "poisson",
        "random_state": random_state,
        "verbose": -1,
    }


class ModelTrainer:
    def __init__(
        self,
        random_state: int = 42,
        test_fraction: float = 0.2,
        optuna_trials: int = 30,
        optuna_timeout_seconds: int = 600,
    ):
        self.random_state = random_state
        self.test_fraction = test_fraction
        self.optuna_trials = optuna_trials
        self.optuna_timeout_seconds = optuna_timeout_seconds

    def train(
        self,
        dataset: pd.DataFrame,
    ) -> tuple[LGBMRegressor, LGBMRegressor, TrainingReport, EvaluationBundle]:
        if dataset.empty:
            raise ValueError("Dataset is empty. Run fetch_matches first.")

        logger.info("[1/4] Разбиение данных (temporal split)...")
        train_df, test_df = temporal_split(dataset, self.test_fraction)
        features_train = train_df[FEATURE_COLUMNS].fillna(0.0)
        features_test = test_df[FEATURE_COLUMNS].fillna(0.0)

        if features_test.empty:
            features_test = features_train
            test_df = train_df

        logger.info("  train=%s, test=%s", len(features_train), len(features_test))

        logger.info("[2/4] Подбор гиперпараметров...")
        if self.optuna_trials > 0 and len(features_train) >= 50:
            home_params = _tune_poisson_regressor(
                features_train,
                train_df["score_home"],
                label="λ хозяев",
                random_state=self.random_state,
                n_trials=self.optuna_trials,
                timeout_seconds=self.optuna_timeout_seconds,
            )
            away_params = _tune_poisson_regressor(
                features_train,
                train_df["score_away"],
                label="λ гостей",
                random_state=self.random_state,
                n_trials=max(10, self.optuna_trials // 2),
                timeout_seconds=self.optuna_timeout_seconds // 2,
            )
        else:
            logger.info("  Optuna пропущен (trials=%s, train=%s)", self.optuna_trials, len(features_train))
            home_params = {**DEFAULT_POISSON_PARAMS, "random_state": self.random_state, "verbose": -1}
            away_params = {**DEFAULT_POISSON_PARAMS, "random_state": self.random_state, "verbose": -1}

        logger.info("[3/4] Финальное обучение Poisson-регрессоров...")
        home_score_model = LGBMRegressor(**home_params)
        away_score_model = LGBMRegressor(**away_params)

        _fit_regressor(
            home_score_model,
            features_train,
            train_df["score_home"],
            label="λ хозяев",
        )
        _fit_regressor(
            away_score_model,
            features_train,
            train_df["score_away"],
            label="λ гостей",
        )

        logger.info("[4/4] Оценка на тесте...")
        home_pred = home_score_model.predict(features_test)
        away_pred = away_score_model.predict(features_test)
        outcome_pred, outcome_proba = predict_outcomes_batch(home_pred, away_pred)

        outcome_accuracy = float(accuracy_score(test_df["result"], outcome_pred))
        home_mae = float(mean_absolute_error(test_df["score_home"], home_pred))
        away_mae = float(mean_absolute_error(test_df["score_away"], away_pred))
        logger.info(
            "  точность исхода=%.1f%%, MAE λ: хозяева=%.2f, гости=%.2f",
            outcome_accuracy * 100,
            home_mae,
            away_mae,
        )

        evaluation = EvaluationBundle(
            train_df=train_df,
            test_df=test_df,
            outcome_pred=outcome_pred,
            home_pred=home_pred,
            away_pred=away_pred,
            outcome_proba=outcome_proba,
            outcome_classes=list(OUTCOME_CLASSES),
        )

        report = TrainingReport(
            outcome_accuracy=outcome_accuracy,
            home_score_mae=home_mae,
            away_score_mae=away_mae,
            train_size=len(features_train),
            test_size=len(features_test),
            models_dir=Path(),
            league_id=0,
            league_name=None,
            split_type="temporal",
            model_type=MODEL_TYPE,
            best_params={"home_regressor": home_params, "away_regressor": away_params},
        )
        return home_score_model, away_score_model, report, evaluation


def train_and_save(
    dataset: pd.DataFrame,
    models_dir: Path,
    *,
    league_id: int,
    league_name: str | None,
    random_state: int = 42,
    test_fraction: float = 0.2,
    optuna_trials: int = 30,
    optuna_timeout_seconds: int = 600,
    reports_dir: Path | None = None,
    generate_plots: bool = True,
) -> TrainingReport:
    models_dir.mkdir(parents=True, exist_ok=True)
    total_started = time.perf_counter()

    trainer = ModelTrainer(
        random_state=random_state,
        test_fraction=test_fraction,
        optuna_trials=optuna_trials,
        optuna_timeout_seconds=optuna_timeout_seconds,
    )
    home_model, away_model, report, evaluation = trainer.train(dataset)

    logger.info("Сохранение моделей в %s...", models_dir)
    joblib.dump(home_model, models_dir / "home_score_model.joblib")
    joblib.dump(away_model, models_dir / "away_score_model.joblib")
    joblib.dump(FEATURE_COLUMNS, models_dir / "feature_columns.joblib")
    joblib.dump(
        {
            "league_id": league_id,
            "league_name": league_name,
            "split_type": "temporal",
            "model_type": MODEL_TYPE,
        },
        models_dir / "model_meta.joblib",
    )
    save_env_info(models_dir)

    legacy_classifier = models_dir / "outcome_model.joblib"
    if legacy_classifier.exists():
        legacy_classifier.unlink()
        logger.info("Удалён устаревший outcome_model.joblib")

    report_paths: list[Path] = []
    resolved_reports_dir = reports_dir
    if generate_plots and resolved_reports_dir is not None:
        logger.info("Генерация отчётов в %s...", resolved_reports_dir)
        report_paths = generate_training_reports(
            home_score_model=home_model,
            away_score_model=away_model,
            evaluation=evaluation,
            reports_dir=resolved_reports_dir,
            league_id=league_id,
            league_name=league_name,
            outcome_accuracy=report.outcome_accuracy,
            home_score_mae=report.home_score_mae,
            away_score_mae=report.away_score_mae,
            train_size=report.train_size,
            test_size=report.test_size,
        )
        logger.info("Отчёты готовы: %s файлов", len(report_paths))

    logger.info("Обучение завершено за %.1f с", time.perf_counter() - total_started)

    return TrainingReport(
        outcome_accuracy=report.outcome_accuracy,
        home_score_mae=report.home_score_mae,
        away_score_mae=report.away_score_mae,
        train_size=report.train_size,
        test_size=report.test_size,
        models_dir=models_dir,
        league_id=league_id,
        league_name=league_name,
        split_type=report.split_type,
        model_type=MODEL_TYPE,
        best_params=report.best_params,
        report_paths=report_paths,
        reports_dir=resolved_reports_dir,
    )
