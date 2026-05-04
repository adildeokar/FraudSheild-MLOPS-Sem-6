"""
Heuristic triggers for automatic retraining suggestions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy import func

from app.core.config import settings
from app.db.database import SessionLocal
from app.db.models import Prediction
from app.ml.drift import compute_drift_vs_reference


def _production_roc_auc() -> float:
    prod = Path(settings.MODELS_DIR) / "production.json"
    if not prod.exists():
        return 0.0
    with open(prod, encoding="utf-8") as f:
        info = json.load(f)
    return float(info.get("metrics", {}).get("roc_auc", 0.0))


def evaluate_smart_retrain() -> Dict[str, Any]:
    reasons: List[str] = []
    drift_info = compute_drift_vs_reference()
    if drift_info.get("drift_detected"):
        reasons.append("feature_drift_detected")

    prod_auc = _production_roc_auc()
    if prod_auc > 0 and prod_auc < 0.85:
        reasons.append("low_production_roc_auc")

    db = SessionLocal()
    try:
        total = db.query(func.count(Prediction.id)).scalar() or 0
        fraud = db.query(func.count(Prediction.id)).filter(Prediction.is_fraud == True).scalar() or 0  # noqa: E712
        rate = fraud / max(total, 1)
        if total > 200 and (rate > 0.25 or rate < 0.001):
            reasons.append("live_fraud_rate_shift")
    finally:
        db.close()

    should = len(reasons) > 0
    return {
        "should_retrain": should,
        "reasons": reasons,
        "drift": drift_info,
        "production_roc_auc": prod_auc,
    }
