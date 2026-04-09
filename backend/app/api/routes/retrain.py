"""
Retraining API Route — auto-retraining pipeline endpoints.
"""

import os
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.core.config import settings

router = APIRouter()

# Global retraining status (simple in-memory state)
retrain_status = {
    "running": False,
    "last_result": None,
    "error": None
}


class RetrainResponse(BaseModel):
    status: str
    improved: bool
    improvement: float
    message: str
    old_model: dict
    new_model: dict
    all_model_metrics: dict


def _run_retrain(db_session, new_data_path: str):
    """Background retraining task."""
    from app.ml.retrain import retrain_with_new_data
    from app.main import reload_model_cache

    retrain_status["running"] = True
    retrain_status["error"] = None
    try:
        result = retrain_with_new_data(new_data_path, db=db_session)
        retrain_status["last_result"] = result
        if result["improved"]:
            reload_model_cache()
            print("[Retrain] Model cache reloaded with new production model")
    except Exception as e:
        retrain_status["error"] = str(e)
        print(f"[Retrain] Error: {e}")
    finally:
        retrain_status["running"] = False
        try:
            db_session.close()
        except Exception:
            pass


@router.post("/retrain", tags=["Retraining"])
async def trigger_retrain(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Trigger the auto-retraining pipeline.
    Loads New_transactions.csv, retrains all models,
    and promotes to production ONLY if performance improves.
    """
    if retrain_status["running"]:
        raise HTTPException(status_code=409, detail="Retraining is already in progress")

    new_data_path = settings.NEW_DATA_FILE
    if not os.path.exists(new_data_path):
        raise HTTPException(
            status_code=404,
            detail=f"New transaction data not found at {new_data_path}. "
                   "Please ensure New_transactions.csv exists in the data directory."
        )

    # Run synchronously for demo (instant response with results)
    from app.ml.retrain import retrain_with_new_data
    from app.main import reload_model_cache

    retrain_status["running"] = True
    retrain_status["error"] = None
    try:
        result = retrain_with_new_data(new_data_path, db=db)
        retrain_status["last_result"] = result
        if result["improved"]:
            reload_model_cache()
        return result
    except Exception as e:
        retrain_status["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        retrain_status["running"] = False


@router.get("/retrain/status", tags=["Retraining"])
async def get_retrain_status():
    """Get the current status of the retraining pipeline."""
    return {
        "running": retrain_status["running"],
        "last_result": retrain_status["last_result"],
        "error": retrain_status["error"],
        "new_data_available": os.path.exists(settings.NEW_DATA_FILE)
    }


@router.get("/retrain/check-data", tags=["Retraining"])
async def check_new_data():
    """Check if new transaction data is available for retraining."""
    path = settings.NEW_DATA_FILE
    exists = os.path.exists(path)
    size = 0
    if exists:
        import pandas as pd
        try:
            df = pd.read_csv(path)
            size = len(df)
        except Exception:
            size = -1
    return {
        "new_data_available": exists,
        "file_path": path,
        "record_count": size
    }
