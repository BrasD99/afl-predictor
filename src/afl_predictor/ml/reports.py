from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from afl_predictor.ml.constants import FEATURE_COLUMNS
from afl_predictor.ml.poisson import OUTCOME_CLASSES

logger = logging.getLogger(__name__)

RESULT_LABELS = {"H": "Хозяева", "D": "Ничья", "A": "Гости"}


@dataclass(frozen=True)
class EvaluationBundle:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    outcome_pred: np.ndarray
    home_pred: np.ndarray
    away_pred: np.ndarray
    outcome_proba: np.ndarray | None
    outcome_classes: list[str] | None


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
        }
    )
    return plt


def _importance_plot(
    plt,
    model,
    *,
    title: str,
    output_path: Path,
    top_n: int = 20,
) -> None:
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    names = [FEATURE_COLUMNS[i] for i in indices]
    values = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.28)))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values[::-1], color="#2563eb", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[::-1], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Важность (gain)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _score_scatter_plot(
    plt,
    actual: np.ndarray,
    predicted: np.ndarray,
    *,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, predicted, alpha=0.65, color="#0d9488", edgecolors="white", linewidths=0.4)
    max_val = max(float(actual.max()), float(predicted.max()), 1.0)
    ax.plot([0, max_val], [0, max_val], "--", color="#94a3b8", linewidth=1.2, label="Идеальный прогноз")
    ax.set_xlim(-0.5, max_val + 0.5)
    ax.set_ylim(-0.5, max_val + 0.5)
    ax.set_xlabel("Факт")
    ax.set_ylabel("Прогноз")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _residuals_plot(
    plt,
    home_actual: np.ndarray,
    home_pred: np.ndarray,
    away_actual: np.ndarray,
    away_pred: np.ndarray,
    output_path: Path,
) -> None:
    home_res = home_actual - home_pred
    away_res = away_actual - away_pred

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(home_res, bins=min(15, max(5, len(home_res) // 3)), color="#2563eb", alpha=0.8)
    axes[0].axvline(0, color="#64748b", linestyle="--", linewidth=1)
    axes[0].set_title("Ошибка: голы хозяев")
    axes[0].set_xlabel("Факт − прогноз")

    axes[1].hist(away_res, bins=min(15, max(5, len(away_res) // 3)), color="#7c3aed", alpha=0.8)
    axes[1].axvline(0, color="#64748b", linestyle="--", linewidth=1)
    axes[1].set_title("Ошибка: голы гостей")
    axes[1].set_xlabel("Факт − прогноз")

    fig.suptitle("Распределение ошибок прогноза счёта", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _metrics_summary_plot(
    plt,
    *,
    outcome_accuracy: float,
    home_mae: float,
    away_mae: float,
    train_size: int,
    test_size: int,
    league_name: str | None,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["Точность исхода", "MAE хозяева", "MAE гости"]
    values = [outcome_accuracy * 100, home_mae, away_mae]
    colors = ["#16a34a", "#2563eb", "#7c3aed"]
    bars = ax.bar(labels, values, color=colors, alpha=0.85)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set_ylabel("Значение (%, для MAE — голы)")
    title = f"Метрики на тесте ({league_name})" if league_name else "Метрики на тесте"
    ax.set_title(title)
    ax.text(
        0.02,
        0.98,
        f"Train: {train_size}  |  Test: {test_size}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _confusion_matrix_plot(
    plt,
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    labels = ["H", "D", "A"]
    present = [label for label in labels if label in set(y_true) | set(y_pred)]
    if not present:
        return

    cm = confusion_matrix(y_true, y_pred, labels=present)
    display_labels = [RESULT_LABELS.get(label, label) for label in present]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Матрица ошибок: исход (Poisson)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _outcome_calibration_plot(
    plt,
    y_true: pd.Series,
    proba: np.ndarray,
    classes: list[str],
    output_path: Path,
) -> None:
    if proba is None or len(proba) == 0:
        return

    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    confidences: list[float] = []
    correct: list[int] = []

    for true_label, row in zip(y_true, proba, strict=True):
        if true_label not in class_to_idx:
            continue
        pred_idx = int(np.argmax(row))
        confidences.append(float(row[pred_idx]))
        correct.append(1 if classes[pred_idx] == true_label else 0)

    if len(confidences) < 5:
        return

    bins = np.linspace(0, 1, 6)
    bin_ids = np.digitize(confidences, bins) - 1
    mean_conf: list[float] = []
    mean_acc: list[float] = []

    for bin_idx in range(len(bins) - 1):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        mean_conf.append(float(np.mean(np.array(confidences)[mask])))
        mean_acc.append(float(np.mean(np.array(correct)[mask])))

    if not mean_conf:
        return

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#94a3b8", label="Идеальная калибровка")
    ax.plot(mean_conf, mean_acc, "o-", color="#2563eb", linewidth=2, markersize=7, label="Модель")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Уверенность модели")
    ax.set_ylabel("Доля верных прогнозов")
    ax.set_title("Калибровка вероятностей исхода")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def generate_training_reports(
    *,
    home_score_model: LGBMRegressor,
    away_score_model: LGBMRegressor,
    evaluation: EvaluationBundle,
    reports_dir: Path,
    league_id: int,
    league_name: str | None,
    outcome_accuracy: float,
    home_score_mae: float,
    away_score_mae: float,
    train_size: int,
    test_size: int,
) -> list[Path]:
    """Сохраняет PNG-графики и metrics.json для вставки в отчёты."""
    plt = _setup_matplotlib()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = reports_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    latest_link = reports_dir / "latest"
    if latest_link.exists() or latest_link.is_symlink():
        if latest_link.is_symlink():
            latest_link.unlink()
        elif latest_link.is_dir():
            # Не удаляем старые отчёты — только перезаписываем symlink при наличии.
            pass

    try:
        latest_link.symlink_to(run_dir.name, target_is_directory=True)
    except OSError:
        # Windows / sandbox без symlink — копируем путь в текстовый файл.
        (reports_dir / "latest.txt").write_text(str(run_dir), encoding="utf-8")

    test_df = evaluation.test_df
    paths: list[Path] = []

    plot_specs: list[tuple[str, callable]] = [
        (
            "01_metrics_summary.png",
            lambda p: _metrics_summary_plot(
                plt,
                outcome_accuracy=outcome_accuracy,
                home_mae=home_score_mae,
                away_mae=away_score_mae,
                train_size=train_size,
                test_size=test_size,
                league_name=league_name,
                output_path=p,
            ),
        ),
        (
            "02_confusion_matrix.png",
            lambda p: _confusion_matrix_plot(
                plt, test_df["result"], evaluation.outcome_pred, p
            ),
        ),
        (
            "03_feature_importance_home_score.png",
            lambda p: _importance_plot(
                plt,
                home_score_model,
                title="Важность признаков: λ голов хозяев (Poisson)",
                output_path=p,
            ),
        ),
        (
            "04_feature_importance_away_score.png",
            lambda p: _importance_plot(
                plt,
                away_score_model,
                title="Важность признаков: λ голов гостей (Poisson)",
                output_path=p,
            ),
        ),
        (
            "05_score_scatter_home.png",
            lambda p: _score_scatter_plot(
                plt,
                test_df["score_home"].to_numpy(),
                evaluation.home_pred,
                title="Голы хозяев: факт vs λ",
                output_path=p,
            ),
        ),
        (
            "06_score_scatter_away.png",
            lambda p: _score_scatter_plot(
                plt,
                test_df["score_away"].to_numpy(),
                evaluation.away_pred,
                title="Голы гостей: факт vs λ",
                output_path=p,
            ),
        ),
        (
            "07_score_residuals.png",
            lambda p: _residuals_plot(
                plt,
                test_df["score_home"].to_numpy(),
                evaluation.home_pred,
                test_df["score_away"].to_numpy(),
                evaluation.away_pred,
                p,
            ),
        ),
    ]

    for index, (filename, draw) in enumerate(plot_specs, start=1):
        path = run_dir / filename
        logger.info("  [%s/%s] %s", index, len(plot_specs), filename)
        draw(path)
        paths.append(path)

    if evaluation.outcome_proba is not None and len(evaluation.outcome_proba) > 0:
        cal_path = run_dir / "08_outcome_calibration.png"
        logger.info("  [%s/%s] %s", len(plot_specs) + 1, len(plot_specs) + 1, cal_path.name)
        _outcome_calibration_plot(
            plt,
            test_df["result"],
            evaluation.outcome_proba,
            list(evaluation.outcome_classes or OUTCOME_CLASSES),
            cal_path,
        )
        paths.append(cal_path)

    metrics: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "poisson_score",
        "league_id": league_id,
        "league_name": league_name,
        "train_size": train_size,
        "test_size": test_size,
        "outcome_accuracy": round(outcome_accuracy, 4),
        "outcome_accuracy_pct": round(outcome_accuracy * 100, 1),
        "home_score_mae": round(home_score_mae, 3),
        "away_score_mae": round(away_score_mae, 3),
        "plots": [p.name for p in paths],
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.append(metrics_path)

    readme = run_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                "Отчёт обучения AFL Predictor (Poisson)",
                f"Лига: {league_name or league_id}",
                f"Точность исхода (из λ): {outcome_accuracy * 100:.1f}%",
                f"MAE λ хозяева / гости: {home_score_mae:.2f} / {away_score_mae:.2f}",
                "",
                "Файлы для отчётов:",
                "  01_metrics_summary.png      — сводка метрик",
                "  02_confusion_matrix.png     — матрица ошибок H/D/A (Poisson)",
                "  03-04_feature_importance_*  — важность признаков регрессоров",
                "  05-06_score_scatter_*       — факт vs λ",
                "  07_score_residuals.png      — распределение ошибок λ",
                "  08_outcome_calibration.png  — калибровка вероятностей",
                "  metrics.json                — метрики в JSON",
            ]
        ),
        encoding="utf-8",
    )
    paths.append(readme)

    return paths
