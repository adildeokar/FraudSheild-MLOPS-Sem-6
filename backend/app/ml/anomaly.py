"""
Unsupervised Isolation Forest for zero-day / unknown-pattern anomalies.
Trained on normal (Class==0) transactions from the active dataset.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

from app.core.config import settings
from app.core.data_source import get_active_training_data_path
from app.ml.pipeline import FEATURE_COLS, load_data, preprocess


_ANOMALY_MODEL: Tuple[Optional[IsolationForest], Optional[Any]] = (None, None)


def _anomaly_cache_path() -> Path:
    return Path(settings.MODELS_DIR) / "isolation_forest_anomaly.joblib"


def train_anomaly_detector(force: bool = False) -> str:
    """Fit IsolationForest on normal-only data; persist to models directory."""
    path = _anomaly_cache_path()
    if path.exists() and not force:
        return str(path)

    data_path = get_active_training_data_path()
    df = load_data(data_path)
    normal = df[df["Class"] == 0]
    if len(normal) < 200:
        normal = df  # fallback if too few normal rows
    X = normal[FEATURE_COLS]
    X_train, _ = train_test_split(X, test_size=0.2, random_state=42)
    X_scaled, scaler = preprocess(X_train, fit=True)

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.02,
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(X_scaled.values)

    bundle = {"model": iso, "scaler": scaler}
    os.makedirs(settings.MODELS_DIR, exist_ok=True)
    joblib.dump(bundle, path)
    return str(path)


def load_anomaly_bundle() -> dict:
    global _ANOMALY_MODEL
    path = _anomaly_cache_path()
    if not path.exists():
        train_anomaly_detector(force=True)
    bundle = joblib.load(path)
    _ANOMALY_MODEL = (bundle["model"], bundle["scaler"])
    return bundle


def anomaly_score_for_features(feature_frame: pd.DataFrame) -> dict:
    bundle = load_anomaly_bundle()
    iso: IsolationForest = bundle["model"]
    scaler = bundle["scaler"]
    X = feature_frame[FEATURE_COLS].copy()
    Xs, _ = preprocess(X, scaler=scaler, fit=False)
    # decision_function: lower = more anomalous (more negative)
    d = iso.decision_function(Xs.values).ravel()
    # Normalize to [0,1] score where 1 = more anomalous
    # Typical range ~[-0.5, 0.5]; invert and min-max on batch of 1 row → use logistic-style mapping
    raw = float(d[0])
    score = float(1.0 / (1.0 + np.exp(raw * 5.0)))
    pred = iso.predict(Xs.values)[0]
    is_anomaly = pred == -1
    return {
        "anomaly_score": round(score, 6),
        "is_anomaly": bool(is_anomaly),
        "isolation_raw": round(raw, 6),
    }
