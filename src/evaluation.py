"""Evaluation helpers for real-estate regression models."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def evaluate_regression_predictions(
    model_name: str,
    y_true: Iterable[float],
    y_pred: Iterable[float],
) -> dict[str, float | str]:
    """Return standard regression metrics for model predictions."""
    y_true_values = list(y_true)
    y_pred_values = list(y_pred)
    if not y_true_values or not y_pred_values:
        raise ValueError("Regression evaluation requires at least one prediction.")
    if len(y_true_values) != len(y_pred_values):
        raise ValueError("y_true and y_pred must have the same length.")

    return {
        "model": model_name,
        "mae": float(mean_absolute_error(y_true_values, y_pred_values)),
        "rmse": float(root_mean_squared_error(y_true_values, y_pred_values)),
        "r2": float(r2_score(y_true_values, y_pred_values)),
    }


def compare_regression_results(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Create a ranked metrics table sorted by lowest MAE."""
    if not results:
        raise ValueError("At least one model result is required.")

    table = pd.DataFrame(results).sort_values("mae", ascending=True).reset_index(drop=True)
    table["rank_by_mae"] = range(1, len(table) + 1)
    return table
