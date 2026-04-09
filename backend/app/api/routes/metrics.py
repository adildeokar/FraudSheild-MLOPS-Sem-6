"""
Metrics & Analytics API Routes — model performance and statistics.
"""

import json
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from app.db.database import get_db
from app.db.models import Prediction, ModelRegistry
from app.core.config import settings

router = APIRouter()


@router.get("/health", tags=["System"])
async def health_check():
    """System health check."""
    from app.main import model_cache
    return {
        "status": "healthy",
        "model_loaded": model_cache["model"] is not None,
        "model_version": model_cache.get("version", "none"),
        "model_type": model_cache.get("model_type", "none"),
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/metrics", tags=["Metrics"])
async def get_current_metrics(db: Session = Depends(get_db)):
    """Return current production model performance metrics."""
    from app.main import model_cache
    from pathlib import Path

    prod_file = Path(settings.MODELS_DIR) / "production.json"
    if not prod_file.exists():
        return {"error": "No production model found"}

    with open(prod_file) as f:
        info = json.load(f)

    total_preds = db.query(func.count(Prediction.id)).scalar() or 0
    fraud_preds = db.query(func.count(Prediction.id)).filter(Prediction.is_fraud == True).scalar() or 0

    return {
        "version": info.get("version"),
        "model_type": info.get("model_type"),
        "metrics": info.get("metrics", {}),
        "deployed_at": info.get("set_at"),
        "total_predictions": total_preds,
        "fraud_detected": fraud_preds,
        "fraud_rate": round(fraud_preds / max(total_preds, 1) * 100, 2)
    }


@router.get("/stats", tags=["Analytics"])
async def get_fraud_stats(db: Session = Depends(get_db)):
    """Return fraud detection statistics."""
    total = db.query(func.count(Prediction.id)).scalar() or 0
    fraud = db.query(func.count(Prediction.id)).filter(Prediction.is_fraud == True).scalar() or 0
    normal = total - fraud

    avg_fraud_prob = db.query(func.avg(Prediction.fraud_probability)).scalar() or 0.0
    avg_proc_time = db.query(func.avg(Prediction.processing_time_ms)).scalar() or 0.0

    risk_counts = (
        db.query(Prediction.risk_level, func.count(Prediction.id))
        .group_by(Prediction.risk_level)
        .all()
    )
    risk_distribution = {level: count for level, count in risk_counts}

    # Last 7 days daily counts
    daily_stats = []
    for i in range(7):
        day = datetime.utcnow() - timedelta(days=6 - i)
        day_start = day.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day.replace(hour=23, minute=59, second=59)
        day_total = db.query(func.count(Prediction.id)).filter(
            Prediction.created_at >= day_start,
            Prediction.created_at <= day_end
        ).scalar() or 0
        day_fraud = db.query(func.count(Prediction.id)).filter(
            Prediction.is_fraud == True,
            Prediction.created_at >= day_start,
            Prediction.created_at <= day_end
        ).scalar() or 0
        daily_stats.append({
            "date": day.strftime("%Y-%m-%d"),
            "total": day_total,
            "fraud": day_fraud,
            "normal": day_total - day_fraud
        })

    return {
        "total_predictions": total,
        "total_fraud": fraud,
        "total_normal": normal,
        "fraud_rate_percent": round(fraud / max(total, 1) * 100, 2),
        "avg_fraud_probability": round(float(avg_fraud_prob), 4),
        "avg_processing_time_ms": round(float(avg_proc_time), 2),
        "risk_distribution": risk_distribution,
        "daily_stats": daily_stats
    }


@router.get("/history", tags=["Analytics"])
async def get_prediction_history(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    fraud_only: bool = Query(False),
    db: Session = Depends(get_db)
):
    """Return recent prediction history."""
    query = db.query(Prediction)
    if fraud_only:
        query = query.filter(Prediction.is_fraud == True)
    predictions = query.order_by(desc(Prediction.created_at)).offset(offset).limit(limit).all()

    return {
        "predictions": [
            {
                "id": p.id,
                "transaction_id": p.transaction_id,
                "is_fraud": p.is_fraud,
                "fraud_probability": p.fraud_probability,
                "risk_level": p.risk_level,
                "model_version": p.model_version,
                "model_type": p.model_type,
                "processing_time_ms": p.processing_time_ms,
                "amount": p.amount,
                "created_at": p.created_at.isoformat() if p.created_at else None
            }
            for p in predictions
        ],
        "total": len(predictions),
        "offset": offset,
        "limit": limit
    }


@router.get("/model-registry", tags=["Metrics"])
async def get_model_registry(db: Session = Depends(get_db)):
    """Return all model versions from the registry."""
    models = db.query(ModelRegistry).order_by(desc(ModelRegistry.created_at)).all()

    registry = []
    for m in models:
        notes = m.comparison_notes or ""
        all_metrics = {}
        if "All models:" in notes:
            try:
                json_str = notes.split("All models:")[1].strip()
                all_metrics = json.loads(json_str)
            except Exception:
                pass

        registry.append({
            "id": m.id,
            "version": m.version,
            "model_type": m.model_type,
            "accuracy": m.accuracy,
            "precision": m.precision,
            "recall": m.recall,
            "f1_score": m.f1_score,
            "roc_auc": m.roc_auc,
            "confusion_matrix": {
                "tn": m.true_negatives,
                "fp": m.false_positives,
                "fn": m.false_negatives,
                "tp": m.true_positives
            },
            "dataset_size": m.dataset_size,
            "fraud_count": m.fraud_count,
            "is_production": m.is_production,
            "comparison_notes": notes,
            "all_model_metrics": all_metrics,
            "created_at": m.created_at.isoformat() if m.created_at else None
        })

    return {"models": registry, "total": len(registry)}


@router.get("/feature-importance", tags=["Metrics"])
async def get_feature_importance():
    """Return feature importance for the current production model."""
    from app.main import model_cache
    from app.ml.pipeline import get_feature_importance, FEATURE_COLS

    if model_cache["model"] is None:
        return {"error": "No model loaded"}

    importance = get_feature_importance(model_cache["model"], model_cache.get("model_type", ""))
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    return {
        "model_version": model_cache.get("version"),
        "model_type": model_cache.get("model_type"),
        "feature_importance": [
            {"feature": k, "importance": v} for k, v in sorted_importance
        ]
    }
