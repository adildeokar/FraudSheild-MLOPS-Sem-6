"""
Load and align real-world credit card fraud datasets (Kaggle / OpenML) with the pipeline schema.

Optional IEEE-CIS: only accepted if the CSV already contains Time, V1–V28, Amount, Class
(a reduced export). Otherwise users should upload a compatible CSV via /api/data/upload.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from app.core.config import settings
from app.ml.pipeline import FEATURE_COLS

# Default OpenML id: Credit Card Fraud (same statistical structure as Kaggle European 2013)
OPENML_CREDITCARD_ID = 1597

REQUIRED_TARGET = "Class"
# Minimum columns after we synthesize Time if absent
BASE_FEATURE_COLS = FEATURE_COLS


def _ensure_time_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Time" in df.columns:
        return df
    out = df.copy()
    # Preserve row order as pseudo-time when the source omits Time (e.g. some OpenML dumps)
    out["Time"] = np.arange(len(out), dtype=float) * 0.01
    return out


def _normalize_class(series: pd.Series) -> pd.Series:
    s = series.copy()
    if s.dtype.name == "category" or s.dtype == object:
        s = pd.to_numeric(s.astype(str), errors="coerce")
    return s.fillna(0).astype(int)


def preprocess_real_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop invalid rows, coerce types, ensure Class in {0,1}."""
    df = df.copy()
    for c in BASE_FEATURE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if REQUIRED_TARGET in df.columns:
        df[REQUIRED_TARGET] = _normalize_class(df[REQUIRED_TARGET])
    df = df.dropna(subset=[c for c in BASE_FEATURE_COLS if c in df.columns] + ([REQUIRED_TARGET] if REQUIRED_TARGET in df.columns else []))
    df = df.reset_index(drop=True)
    return df


def align_features_with_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder / trim columns to match training FEATURE_COLS + Class."""
    df = _ensure_time_column(df)
    missing = [c for c in FEATURE_COLS + [REQUIRED_TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot align dataset; missing columns: {missing}")
    return df[FEATURE_COLS + [REQUIRED_TARGET]].copy()


def load_kaggle_dataset(local_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load the European credit-card fraud dataset: local CSV (Kaggle) or OpenML if no file.
    Environment REAL_DATA_MAX_ROWS limits size for faster training (default 80000).
    """
    max_rows = int(os.environ.get("REAL_DATA_MAX_ROWS", str(settings.REAL_DATA_MAX_ROWS)))

    if local_path and os.path.isfile(local_path):
        df = pd.read_csv(local_path)
    else:
        kaggle_default = Path(settings.DATA_DIR) / "kaggle_creditcard_source.csv"
        if kaggle_default.is_file():
            df = pd.read_csv(kaggle_default)
        else:
            from sklearn.datasets import fetch_openml

            bunch = fetch_openml(data_id=OPENML_CREDITCARD_ID, as_frame=True, parser="auto")
            Xs = bunch.data
            ys = pd.to_numeric(bunch.target.astype(str), errors="coerce").fillna(0).astype(int)
            df = Xs.copy()
            df[REQUIRED_TARGET] = ys.values

    df = preprocess_real_data(df)
    df = align_features_with_pipeline(df)

    if max_rows > 0 and len(df) > max_rows:
        # Stratified sample to keep fraud cases
        fraud = df[df[REQUIRED_TARGET] == 1]
        norm = df[df[REQUIRED_TARGET] == 0]
        frac_f = max(0.02, len(fraud) / max(len(df), 1))
        n_fraud = min(len(fraud), max(int(max_rows * frac_f), 50))
        n_norm = max_rows - n_fraud
        fraud_s = fraud.sample(n=min(n_fraud, len(fraud)), random_state=42) if len(fraud) else fraud
        norm_s = norm.sample(n=min(n_norm, len(norm)), random_state=42) if len(norm) else norm
        df = pd.concat([fraud_s, norm_s], ignore_index=True).sample(frac=1.0, random_state=42)

    return df


def load_ieee_or_aligned_csv(path: str) -> pd.DataFrame:
    """
    Optional IEEE-CIS-style file: must contain at least Time, V1–V28, Amount, Class.
    """
    df = pd.read_csv(path)
    return preprocess_real_data(align_features_with_pipeline(df))


def save_processed_real(df: pd.DataFrame, out_path: Optional[str] = None) -> str:
    """Write processed real dataset for production use (never overwrites repo-root creditcard.csv)."""
    target = out_path or settings.REAL_OPENML_SAMPLE_PATH
    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    df.to_csv(target, index=False)
    return target


def build_real_creditcard_file(
    local_path: Optional[str] = None,
    out_path: Optional[str] = None,
) -> Tuple[str, int, int]:
    """
    End-to-end: load → preprocess → persist.
    Returns (path, n_rows, fraud_count).
    """
    df = load_kaggle_dataset(local_path=local_path)
    path = save_processed_real(df, out_path=out_path)
    fraud_count = int(df[REQUIRED_TARGET].sum())
    return path, len(df), fraud_count
