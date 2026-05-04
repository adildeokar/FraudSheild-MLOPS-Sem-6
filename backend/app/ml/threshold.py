"""
Optimal probability thresholds: F1 maximization and cost-based.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from app.core.data_source import get_active_training_data_path
from app.ml.pipeline import load_data, preprocess, FEATURE_COLS


def _compute_cost_threshold(y_true: np.ndarray, y_prob: np.ndarray, cost_fp: float, cost_fn: float) -> float:
    grid = np.unique(np.concatenate([[0.05, 0.95], np.quantile(y_prob, np.linspace(0.05, 0.95, 60))]))
    best_t = 0.5
    best_cost = float("inf")
    for t in grid:
        pred = (y_prob >= t).astype(int)
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        c = fp * cost_fp + fn * cost_fn
        if c < best_cost:
            best_cost = c
            best_t = float(t)
    return best_t


def compute_optimal_thresholds(model, scaler, data_path: str | None = None) -> Dict[str, Any]:
    path = data_path or get_active_training_data_path()
    df = load_data(path)
    X = df[FEATURE_COLS]
    y = df["Class"].values
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    X_test_s, _ = preprocess(X_test, scaler=scaler, fit=False)
    y_prob = model.predict_proba(X_test_s.values)[:, 1]

    grid = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_f1 = 0.0
    for t in grid:
        f1 = f1_score(y_test, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_t = float(t)

    baseline_f1 = float(f1_score(y_test, (y_prob >= 0.5).astype(int), zero_division=0))
    improvement = ((best_f1 - baseline_f1) / max(baseline_f1, 1e-6)) * 100.0

    cost_fp, cost_fn = 1.0, 10.0
    t_cost = _compute_cost_threshold(y_test, y_prob, cost_fp, cost_fn)

    return {
        "optimal_threshold": round(best_t, 6),
        "optimal_threshold_cost": round(t_cost, 6),
        "method": "f1_max",
        "f1_at_optimal": round(best_f1, 6),
        "f1_at_0_5": round(baseline_f1, 6),
        "expected_improvement": f"{round(improvement, 2):+.1f}%",
        "expected_improvement_pct": round(improvement, 2),
        "cost_assumptions": {"false_positive": cost_fp, "false_negative": cost_fn},
        "eval_samples": len(y_test),
    }
