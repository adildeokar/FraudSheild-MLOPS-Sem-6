"""
Population Stability Index (PSI) and Kolmogorov–Smirnov drift on key features.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from app.ml.pipeline import FEATURE_COLS, load_data
from app.core.data_source import get_active_training_data_path


def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    PSI for a single feature; clip to avoid log(0).
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 20 or len(actual) < 10:
        return 0.0
    qs = np.linspace(0, 1, buckets + 1)
    breaks = np.unique(np.quantile(expected, qs))
    if len(breaks) < 3:
        breaks = np.linspace(np.min(expected), np.max(expected), buckets + 1)
    e_counts, _ = np.histogram(expected, bins=breaks)
    a_counts, _ = np.histogram(actual, bins=breaks)
    e_pct = e_counts / max(np.sum(e_counts), 1)
    a_pct = a_counts / max(np.sum(a_counts), 1)
    e_pct = np.clip(e_pct, 1e-6, 1.0)
    a_pct = np.clip(a_pct, 1e-6, 1.0)
    psi_vals = (a_pct - e_pct) * np.log(a_pct / e_pct)
    return float(np.sum(psi_vals))


def compute_drift_vs_reference(
    reference_df: pd.DataFrame | None = None,
    current_df: pd.DataFrame | None = None,
    feature_subset: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Compare `current_df` to `reference_df` (defaults: training CSV vs recent window).
    If current_df is None, builds current from last N prediction-linked rows (Amount, V-features simulated from DB is limited).
    """
    ref_path = get_active_training_data_path()
    if reference_df is None:
        reference_df = load_data(ref_path)
    if feature_subset is None:
        feature_subset = ["Amount", "Time", "V1", "V14", "V4", "V10"]

    cols = [c for c in feature_subset if c in reference_df.columns]
    if not cols:
        cols = FEATURE_COLS[:8]

    if current_df is None:
        current_df = _build_current_from_predictions_and_or_training(reference_df)

    psi_scores: List[float] = []
    ks_stats: List[float] = []
    ks_pvals: List[float] = []

    for col in cols:
        ref = reference_df[col].astype(float).values
        cur = current_df[col].astype(float).values
        psi_scores.append(_psi(ref, cur))
        if len(ref) > 10 and len(cur) > 10:
            ks, p = stats.ks_2samp(ref, cur, alternative="two-sided", mode="auto")
            ks_stats.append(float(ks))
            ks_pvals.append(float(p))
        else:
            ks_stats.append(0.0)
            ks_pvals.append(1.0)

    psi_mean = float(np.mean(psi_scores)) if psi_scores else 0.0
    ks_mean = float(np.mean(ks_stats)) if ks_stats else 0.0
    # Heuristic: significant drift if mean PSI > 0.2 or strong KS
    drift_detected = psi_mean > 0.2 or ks_mean > 0.25
    recommendation = "RETRAIN" if drift_detected else "MONITOR"

    return {
        "psi_score": round(psi_mean, 6),
        "ks_stat": round(ks_mean, 6),
        "psi_per_feature": {cols[i]: round(psi_scores[i], 6) for i in range(len(cols))},
        "ks_per_feature": {cols[i]: round(ks_stats[i], 6) for i in range(len(cols))},
        "drift_detected": drift_detected,
        "recommendation": recommendation,
        "reference_rows": len(reference_df),
        "current_rows": len(current_df),
    }


def _build_current_from_predictions_and_or_training(ref_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer recent production transactions from SQLite; otherwise tail of the reference file
    with small Amount jitter as a proxy window.
    """
    try:
        from app.db.database import SessionLocal
        from app.db.models import Transaction

        db = SessionLocal()
        try:
            rows = db.query(Transaction).order_by(Transaction.id.desc()).limit(1200).all()
            if len(rows) >= 80:
                records = []
                for r in rows:
                    records.append({
                        "Time": float(r.time_feature),
                        "Amount": float(r.amount),
                        **{f"V{i}": float(getattr(r, f"v{i}")) for i in range(1, 29)},
                    })
                return pd.DataFrame.from_records(records)
        finally:
            db.close()
    except Exception:
        pass

    n = len(ref_df)
    k = max(int(n * 0.2), 200)
    tail = ref_df.tail(k).copy()
    if "Amount" in tail.columns:
        rng = np.random.default_rng(42)
        tail["Amount"] = tail["Amount"] * (1.0 + rng.normal(0, 0.02, size=len(tail)))
    return tail
