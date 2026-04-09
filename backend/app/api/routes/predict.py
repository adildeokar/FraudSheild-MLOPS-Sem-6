"""
Prediction API Route — real-time fraud detection endpoint.
"""

import time
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Prediction, Transaction
from app.core.config import settings

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class TransactionInput(BaseModel):
    time: float = Field(..., description="Seconds elapsed from first transaction")
    v1: float; v2: float; v3: float; v4: float
    v5: float; v6: float; v7: float; v8: float
    v9: float; v10: float; v11: float; v12: float
    v13: float; v14: float; v15: float; v16: float
    v17: float; v18: float; v19: float; v20: float
    v21: float; v22: float; v23: float; v24: float
    v25: float; v26: float; v27: float; v28: float
    amount: float = Field(..., ge=0, description="Transaction amount in USD")

    class Config:
        json_schema_extra = {
            "example": {
                "time": 0.0, "amount": 149.62,
                "v1": -1.3598, "v2": -0.0728, "v3": 2.5363, "v4": 1.3782,
                "v5": -0.3383, "v6": 0.4624, "v7": 0.2396, "v8": 0.0987,
                "v9": 0.3638, "v10": 0.0908, "v11": -0.5516, "v12": -0.6178,
                "v13": -0.9913, "v14": -0.3112, "v15": 1.4682, "v16": -0.4704,
                "v17": 0.2079, "v18": 0.0258, "v19": 0.4030, "v20": 0.2514,
                "v21": -0.0183, "v22": 0.2778, "v23": -0.1104, "v24": 0.0669,
                "v25": 0.1285, "v26": -0.1891, "v27": 0.1336, "v28": -0.0211
            }
        }


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_level: str
    model_version: str
    model_type: str
    processing_time_ms: float
    timestamp: str


class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionInput]


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total: int
    fraud_detected: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def determine_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "LOW"
    elif probability < 0.6:
        return "MEDIUM"
    elif probability < 0.85:
        return "HIGH"
    else:
        return "CRITICAL"


def make_prediction(transaction: TransactionInput) -> PredictionResponse:
    """Core prediction logic using the globally loaded model."""
    import numpy as np
    from app.main import model_cache

    if model_cache["model"] is None or model_cache["scaler"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train first.")

    model = model_cache["model"]
    scaler = model_cache["scaler"]
    version = model_cache["version"]
    model_type = model_cache["model_type"]

    t0 = time.time()

    features = [
        transaction.time, transaction.v1, transaction.v2, transaction.v3,
        transaction.v4, transaction.v5, transaction.v6, transaction.v7,
        transaction.v8, transaction.v9, transaction.v10, transaction.v11,
        transaction.v12, transaction.v13, transaction.v14, transaction.v15,
        transaction.v16, transaction.v17, transaction.v18, transaction.v19,
        transaction.v20, transaction.v21, transaction.v22, transaction.v23,
        transaction.v24, transaction.v25, transaction.v26, transaction.v27,
        transaction.v28, transaction.amount
    ]

    import pandas as pd
    from app.ml.pipeline import FEATURE_COLS, SCALE_COLS

    X = pd.DataFrame([features], columns=FEATURE_COLS)
    X[SCALE_COLS] = scaler.transform(X[SCALE_COLS])

    fraud_prob = float(model.predict_proba(X.values)[0, 1])
    is_fraud = fraud_prob >= settings.FRAUD_THRESHOLD
    risk_level = determine_risk_level(fraud_prob)
    processing_ms = round((time.time() - t0) * 1000, 2)
    transaction_id = str(uuid.uuid4())

    return PredictionResponse(
        transaction_id=transaction_id,
        is_fraud=is_fraud,
        fraud_probability=round(fraud_prob, 6),
        risk_level=risk_level,
        model_version=version,
        model_type=model_type,
        processing_time_ms=processing_ms,
        timestamp=datetime.utcnow().isoformat()
    ), transaction_id, transaction.amount


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_transaction(transaction: TransactionInput, db: Session = Depends(get_db)):
    """Predict whether a transaction is fraudulent."""
    result, transaction_id, amount = make_prediction(transaction)

    pred_record = Prediction(
        transaction_id=transaction_id,
        is_fraud=result.is_fraud,
        fraud_probability=result.fraud_probability,
        risk_level=result.risk_level,
        model_version=result.model_version,
        model_type=result.model_type,
        processing_time_ms=result.processing_time_ms,
        amount=amount
    )
    db.add(pred_record)

    tx_record = Transaction(
        transaction_id=transaction_id,
        time_feature=transaction.time,
        amount=transaction.amount,
        v1=transaction.v1, v2=transaction.v2, v3=transaction.v3,
        v4=transaction.v4, v5=transaction.v5, v6=transaction.v6,
        v7=transaction.v7, v8=transaction.v8, v9=transaction.v9,
        v10=transaction.v10, v11=transaction.v11, v12=transaction.v12,
        v13=transaction.v13, v14=transaction.v14, v15=transaction.v15,
        v16=transaction.v16, v17=transaction.v17, v18=transaction.v18,
        v19=transaction.v19, v20=transaction.v20, v21=transaction.v21,
        v22=transaction.v22, v23=transaction.v23, v24=transaction.v24,
        v25=transaction.v25, v26=transaction.v26, v27=transaction.v27,
        v28=transaction.v28
    )
    db.add(tx_record)
    db.commit()

    return result


@router.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest, db: Session = Depends(get_db)):
    """Predict fraud for a batch of transactions."""
    results = []
    for tx in request.transactions:
        result, transaction_id, amount = make_prediction(tx)
        pred_record = Prediction(
            transaction_id=transaction_id,
            is_fraud=result.is_fraud,
            fraud_probability=result.fraud_probability,
            risk_level=result.risk_level,
            model_version=result.model_version,
            model_type=result.model_type,
            processing_time_ms=result.processing_time_ms,
            amount=amount
        )
        db.add(pred_record)
        results.append(result)
    db.commit()
    fraud_count = sum(1 for r in results if r.is_fraud)
    return BatchPredictionResponse(
        results=results,
        total=len(results),
        fraud_detected=fraud_count
    )
