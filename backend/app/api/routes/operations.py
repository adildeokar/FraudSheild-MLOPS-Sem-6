"""
Rollback, live stream simulation, A/B testing, and smart-retrain signals.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.routes.predict import TransactionInput, make_prediction
from app.core.config import settings
from app.core.data_source import get_active_training_data_path
from app.db.database import get_db
from app.db.models import ModelRegistry
from app.main import model_cache, reload_model_cache
from app.ml.pipeline import FEATURE_COLS, load_data, set_production_model
from app.ml.smart_retrain import evaluate_smart_retrain

router = APIRouter(tags=["Operations"])

ROLLBACK_LOG = Path(settings.DATA_DIR) / "rollback_events.log"

_stream_lock = asyncio.Lock()
_stream_running = False
_stream_task: Optional[asyncio.Task] = None
_stream_rows: deque = deque(maxlen=500)
_stream_stats: Dict[str, Any] = {"total": 0, "fraud_flagged": 0, "started_at": None}

_ab_models: Dict[str, Any] = {}
_ab_stats = {
    "a": {"n": 0, "fraud": 0, "sum_proba": 0.0},
    "b": {"n": 0, "fraud": 0, "sum_proba": 0.0},
}
_ab_versions: Dict[str, Optional[str]] = {"a": None, "b": None}


class ABTestBody(BaseModel):
    transaction: TransactionInput
    version_a: Optional[str] = None
    version_b: Optional[str] = None


def _append_rollback_log(event: dict) -> None:
    ROLLBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ROLLBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _load_version_artifacts(db: Session, version: str) -> ModelRegistry:
    row = db.query(ModelRegistry).filter(ModelRegistry.version == version).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"Version {version} not found in registry")
    return row


@router.post("/rollback/{version}")
async def rollback_to_version(version: str, db: Session = Depends(get_db)):
    row = _load_version_artifacts(db, version)
    import os

    if not os.path.isfile(row.model_path) or not os.path.isfile(row.scaler_path):
        raise HTTPException(status_code=400, detail="Model files missing on disk")

    metrics = {
        "accuracy": row.accuracy,
        "precision": row.precision,
        "recall": row.recall,
        "f1_score": row.f1_score,
        "roc_auc": row.roc_auc,
        "true_negatives": row.true_negatives,
        "false_positives": row.false_positives,
        "false_negatives": row.false_negatives,
        "true_positives": row.true_positives,
    }
    set_production_model(version, row.model_path, row.scaler_path, row.model_type, metrics)
    db.query(ModelRegistry).update({ModelRegistry.is_production: False})
    row.is_production = True
    db.commit()
    reload_model_cache()
    evt = {"at": datetime.utcnow().isoformat(), "version": version, "type": "rollback"}
    _append_rollback_log(evt)
    return {"ok": True, "production_version": version, "model_type": row.model_type}


async def _stream_loop():
    global _stream_running
    path = get_active_training_data_path()
    try:
        df = load_data(path)
    except Exception:
        _stream_running = False
        return
    feat_iter = itertools.cycle(df[FEATURE_COLS + ["Class"]].to_dict("records"))

    while _stream_running:
        row = next(feat_iter)
        cls = row.pop("Class", 0)
        tx = TransactionInput(
            time=float(row["Time"]),
            amount=float(row["Amount"]),
            **{f"v{i}": float(row[f"V{i}"]) for i in range(1, 29)},
        )
        try:
            res, _, _ = make_prediction(tx)
            evt = {
                "transaction_id": res.transaction_id,
                "fraud_probability": res.fraud_probability,
                "is_fraud": res.is_fraud,
                "risk_level": res.risk_level,
                "true_label": int(cls),
                "ts": datetime.utcnow().isoformat(),
            }
            _stream_rows.appendleft(evt)
            _stream_stats["total"] += 1
            if res.is_fraud:
                _stream_stats["fraud_flagged"] += 1
        except Exception:
            pass
        await asyncio.sleep(0.1)


@router.post("/stream/start")
async def stream_start():
    global _stream_running, _stream_task, _stream_stats
    async with _stream_lock:
        if _stream_running:
            return {"status": "already_running"}
        _stream_rows.clear()
        _stream_stats = {"total": 0, "fraud_flagged": 0, "started_at": datetime.utcnow().isoformat()}
        _stream_running = True
        _stream_task = asyncio.create_task(_stream_loop())
    return {"status": "started", "interval_ms": 100}


@router.post("/stream/stop")
async def stream_stop():
    global _stream_running, _stream_task
    async with _stream_lock:
        _stream_running = False
        if _stream_task:
            _stream_task.cancel()
            _stream_task = None
    return {"status": "stopped"}


@router.get("/stream/status")
async def stream_status():
    recent = list(_stream_rows)[:80]
    rate = (
        _stream_stats["fraud_flagged"] / max(_stream_stats["total"], 1) * 100.0
    )
    return {
        "running": _stream_running,
        "stats": {**_stream_stats, "fraud_flag_rate_pct": round(rate, 3)},
        "recent": recent,
    }


def _ensure_ab_models(db: Session, va: Optional[str], vb: Optional[str]) -> tuple[str, str]:
    rows = db.query(ModelRegistry).order_by(ModelRegistry.id.desc()).all()
    if len(rows) < 1:
        raise HTTPException(status_code=400, detail="Model registry empty")
    if va is None and vb is None:
        if len(rows) < 2:
            va = vb = rows[0].version
        else:
            va, vb = rows[1].version, rows[0].version
    if va is None:
        va = rows[-1].version
    if vb is None:
        vb = rows[0].version

    key = f"{va}:{vb}"
    if _ab_models.get("__key__") != key:
        import joblib
        import os

        a_row = _load_version_artifacts(db, va)
        b_row = _load_version_artifacts(db, vb)
        if not os.path.isfile(a_row.model_path) or not os.path.isfile(b_row.scaler_path):
            raise HTTPException(status_code=400, detail="Model A files missing")
        if not os.path.isfile(b_row.model_path) or not os.path.isfile(b_row.scaler_path):
            raise HTTPException(status_code=400, detail="Model B files missing")
        _ab_models["a"] = (
            joblib.load(a_row.model_path),
            joblib.load(a_row.scaler_path),
            a_row.model_type,
            va,
        )
        _ab_models["b"] = (
            joblib.load(b_row.model_path),
            joblib.load(b_row.scaler_path),
            b_row.model_type,
            vb,
        )
        _ab_models["__key__"] = key
    _ab_versions["a"], _ab_versions["b"] = va, vb
    return va, vb


def _predict_with_bundle(bundle, tx: TransactionInput) -> Dict[str, Any]:
    model, scaler, mtype, version = bundle
    row = [[tx.time, *[getattr(tx, f"v{i}") for i in range(1, 29)], tx.amount]]
    X = pd.DataFrame(row, columns=FEATURE_COLS)
    from app.ml.pipeline import preprocess

    Xs, _ = preprocess(X, scaler=scaler, fit=False)
    proba = float(model.predict_proba(Xs.values)[0, 1])
    return {
        "version": version,
        "model_type": mtype,
        "fraud_probability": round(proba, 6),
        "is_fraud": proba >= settings.FRAUD_THRESHOLD,
    }


@router.post("/ab-test")
async def ab_test(body: ABTestBody, db: Session = Depends(get_db)):
    """
    Compare two registry models on the same transaction; assign random traffic arm for demo (50/50).
    """
    va, vb = _ensure_ab_models(db, body.version_a, body.version_b)
    if _ab_models.get("a") is None or _ab_models.get("b") is None:
        raise HTTPException(status_code=500, detail="Failed to load A/B models")

    pred_a = _predict_with_bundle(_ab_models["a"], body.transaction)
    pred_b = _predict_with_bundle(_ab_models["b"], body.transaction)
    arm = random.choice(["A", "B"])
    chosen = pred_a if arm == "A" else pred_b
    bucket = "a" if arm == "A" else "b"
    _ab_stats[bucket]["n"] += 1
    _ab_stats[bucket]["sum_proba"] += chosen["fraud_probability"]
    if chosen["is_fraud"]:
        _ab_stats[bucket]["fraud"] += 1

    leader = (
        "tie"
        if pred_a["fraud_probability"] == pred_b["fraud_probability"]
        else ("A" if pred_a["fraud_probability"] > pred_b["fraud_probability"] else "B")
    )
    return {
        "arm": arm,
        "routing_note": "50/50 traffic split — this request served by arm " + arm,
        "prediction_a": pred_a,
        "prediction_b": pred_b,
        "higher_fraud_score_arm": leader,
        "versions": {"a": va, "b": vb},
    }


@router.get("/ab-test/stats")
async def ab_test_stats():
    def _avg(side: str) -> float:
        n = _ab_stats[side]["n"]
        if n == 0:
            return 0.0
        return _ab_stats[side]["sum_proba"] / n

    return {
        "versions": dict(_ab_versions),
        "a": {**_ab_stats["a"], "avg_proba": round(_avg("a"), 6)},
        "b": {**_ab_stats["b"], "avg_proba": round(_avg("b"), 6)},
    }


@router.get("/retrain/smart-check")
async def retrain_smart_check():
    return evaluate_smart_retrain()
