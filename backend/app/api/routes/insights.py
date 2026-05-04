"""
Drift, thresholds, SHAP explainability, anomaly scoring, and advanced analytics curves.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split

from app.api.routes.predict import TransactionInput
from app.core.config import settings
from app.core.data_source import get_active_training_data_path
from app.main import model_cache
from app.ml.anomaly import anomaly_score_for_features
from app.ml.drift import compute_drift_vs_reference
from app.ml.explain import explain_instance, sample_background_from_numpy
from app.ml.pipeline import FEATURE_COLS, load_data, preprocess
from app.ml.threshold import compute_optimal_thresholds

router = APIRouter(tags=["ML Insights"])

_executor = ThreadPoolExecutor(max_workers=2)


@router.get("/drift")
async def get_drift():
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, compute_drift_vs_reference)


@router.get("/threshold/optimal")
async def get_optimal_threshold():
    if model_cache["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        lambda: compute_optimal_thresholds(model_cache["model"], model_cache["scaler"]),
    )


@router.post("/explain")
async def post_explain(tx: TransactionInput):
    if model_cache["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    def _work():
        df_path = get_active_training_data_path()
        df = load_data(df_path)
        sample = df[FEATURE_COLS].sample(min(800, len(df)), random_state=42)
        Xs, _ = preprocess(sample, scaler=model_cache["scaler"], fit=False)
        bg = sample_background_from_numpy(Xs.values, 280)
        row = pd.DataFrame(
            [[
                tx.time, *[getattr(tx, f"v{i}") for i in range(1, 29)], tx.amount
            ]],
            columns=FEATURE_COLS,
        )
        return explain_instance(
            model_cache["model"], model_cache["scaler"], model_cache["model_type"] or "", row, background_X=bg
        )

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _work)


@router.post("/anomaly")
async def post_anomaly(tx: TransactionInput):
    row = pd.DataFrame(
        [[tx.time, *[getattr(tx, f"v{i}") for i in range(1, 29)], tx.amount]],
        columns=FEATURE_COLS,
    )
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, lambda: anomaly_score_for_features(row))


def _curves_blocking() -> dict:
    if model_cache["model"] is None:
        return {"error": "no_model"}
    df = load_data(get_active_training_data_path())
    X = df[FEATURE_COLS]
    y = df["Class"].values
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    Xs, _ = preprocess(X_test, scaler=model_cache["scaler"], fit=False)
    prob = model_cache["model"].predict_proba(Xs.values)[:, 1]
    pred_label = (prob >= settings.FRAUD_THRESHOLD).astype(int)
    fpr, tpr, _ = roc_curve(y_test, prob)
    prec, rec, _ = precision_recall_curve(y_test, prob)
    cm = confusion_matrix(y_test, pred_label)
    step = max(1, len(fpr) // 200)
    step_pr = max(1, len(prec) // 200)
    return {
        "roc": {
            "fpr": [round(float(x), 6) for x in fpr[::step]],
            "tpr": [round(float(x), 6) for x in tpr[::step]],
            "auc": round(float(auc(fpr, tpr)), 6),
        },
        "pr": {
            "precision": [round(float(x), 6) for x in prec[::step_pr]],
            "recall": [round(float(x), 6) for x in rec[::step_pr]],
        },
        "confusion_matrix": {
            "matrix": cm.tolist(),
            "labels": [["TN", "FP"], ["FN", "TP"]],
        },
    }


def _heatmap_blocking() -> dict:
    path = get_active_training_data_path()
    df = load_data(path)
    cols = [c for c in ["V1", "V4", "V10", "V14", "Amount", "Time"] if c in df.columns]
    sample = df[cols].sample(min(4000, len(df)), random_state=42)
    corr = sample.corr().values.tolist()
    return {
        "features": cols,
        "correlation": [[round(float(v), 4) for v in row] for row in corr],
    }


@router.get("/analytics/curves")
async def analytics_curves():
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(_executor, _curves_blocking)
    if data.get("error"):
        raise HTTPException(status_code=503, detail="Model not loaded")
    return data


@router.get("/analytics/correlation")
async def analytics_correlation():
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _heatmap_blocking)


@router.get("/analytics/production-matrix")
async def production_confusion_from_file():
    """Confusion matrix numbers persisted with the production model (training eval)."""
    prod = Path(settings.MODELS_DIR) / "production.json"
    if not prod.exists():
        raise HTTPException(status_code=404, detail="No production model")
    with open(prod, encoding="utf-8") as f:
        info = json.load(f)
    m = info.get("metrics", {})
    tn, fp, fn, tp = m.get("true_negatives"), m.get("false_positives"), m.get("false_negatives"), m.get("true_positives")
    return {
        "version": info.get("version"),
        "matrix": [[tn, fp], [fn, tp]],
        "labels": [["TN", "FP"], ["FN", "TP"]],
    }
