"""
SHAP-based feature attributions for the production classifier.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from app.ml.pipeline import FEATURE_COLS, preprocess

def _load_shap():
    try:
        import shap
        return shap
    except ImportError:
        return None


def explain_instance(
    model,
    scaler,
    model_type: str,
    transaction_row: pd.DataFrame,
    background_X: np.ndarray | None = None,
) -> Dict[str, Any]:
    shap = _load_shap()
    if shap is None:
        return {"error": "shap_not_installed", "top_features": [], "shap_values": []}

    X_single = transaction_row[FEATURE_COLS].copy()
    X_single_s, _ = preprocess(X_single, scaler=scaler, fit=False)
    x_vec = X_single_s.values.astype(np.float64)

    if background_X is None:
        rng = np.random.default_rng(42)
        # tiny synthetic background consistent with scale — should be replaced by training sample in routes
        background_X = rng.normal(size=(64, len(FEATURE_COLS)))

    model_t = (model_type or "").lower()
    explainer = None
    if "logistic" in model_t:
        explainer = shap.LinearExplainer(model, background_X)
    elif "forest" in model_t or "xgboost" in model_t or "xgb" in model_t:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(
            lambda z: model.predict_proba(z)[:, 1],
            shap.sample(background_X, min(100, len(background_X))),
        )

    shap_vals = explainer.shap_values(x_vec)
    if isinstance(shap_vals, list):
        shap_vals = np.array(shap_vals[1] if len(shap_vals) > 1 else shap_vals[0])
    sv = np.ravel(np.array(shap_vals))

    pairs = sorted(zip(FEATURE_COLS, sv), key=lambda p: abs(p[1]), reverse=True)[:10]
    top = []
    for name, val in pairs[:5]:
        top.append(
            {
                "feature": name,
                "shap_value": round(float(val), 6),
                "contribution": "increases_fraud_risk" if val > 0 else "decreases_fraud_risk",
            }
        )

    return {
        "top_features": top,
        "shap_values": [
            {"feature": name, "value": round(float(v), 8)} for name, v in zip(FEATURE_COLS, sv)
        ],
        "base_value": getattr(explainer, "expected_value", None),
    }


def sample_background_from_numpy(X_scaled: np.ndarray, max_rows: int = 300) -> np.ndarray:
    if len(X_scaled) <= max_rows:
        return X_scaled
    idx = np.random.default_rng(42).choice(len(X_scaled), size=max_rows, replace=False)
    return X_scaled[idx]
