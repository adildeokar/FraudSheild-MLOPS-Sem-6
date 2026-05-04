"""
Dataset source switching, real-data preparation, and CSV upload.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, Body, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.config import settings
from app.core.data_source import get_active_training_data_path, read_state, write_state

router = APIRouter(tags=["Data"])


def _paths_equal(a: str, b: str) -> bool:
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:
        return os.path.abspath(a) == os.path.abspath(b)


class DataSwitchBody(BaseModel):
    source: Literal["synthetic", "real", "uploaded"]


class PrepareRealBody(BaseModel):
    local_path: Optional[str] = None


@router.get("/data/source")
async def get_data_source():
    st = read_state()
    path = get_active_training_data_path()
    eff = "synthetic"
    if st.get("source") == "real" and (
        _paths_equal(path, settings.REAL_CREDITCARD_FILE)
        or _paths_equal(path, settings.REAL_OPENML_SAMPLE_PATH)
    ):
        eff = "real"
    elif st.get("source") == "uploaded" and "uploaded" in path.lower():
        eff = "uploaded"
    return {
        "source": st.get("source", "synthetic"),
        "effective": eff,
        "training_path": path,
    }


@router.post("/data/switch")
async def switch_data_source(body: DataSwitchBody):
    write_state(body.source)
    return {
        "ok": True,
        "source": body.source,
        "training_path": get_active_training_data_path(),
    }


@router.post("/data/prepare-real")
async def prepare_real_dataset(body: Optional[PrepareRealBody] = Body(None)):
    """Download / load the real European credit-card dataset and write data/real_creditcard.csv (sample)."""
    body = body or PrepareRealBody()
    from data.real_data_loader import build_real_creditcard_file

    try:
        path, rows, frauds = build_real_creditcard_file(
            local_path=body.local_path, out_path=settings.REAL_OPENML_SAMPLE_PATH
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"ok": True, "path": path, "rows": rows, "fraud_count": frauds}


@router.post("/data/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Accept a CSV with Time, V1–V28, Amount, Class and activate it as the uploaded source."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    safe_name = "uploaded_custom.csv"
    dest = os.path.join(settings.DATA_DIR, safe_name)
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    try:
        with open(dest, "wb") as out:
            shutil.copyfileobj(file.file, out)
    finally:
        await file.close()

    # Validate minimal schema
    import pandas as pd

    df = pd.read_csv(dest, nrows=5)
    required = {"Time", "Amount", "Class"} | {f"V{i}" for i in range(1, 29)}
    miss = required - set(df.columns)
    if miss:
        os.remove(dest)
        raise HTTPException(status_code=400, detail=f"Missing columns: {sorted(miss)}")

    write_state("uploaded", upload_filename=safe_name)
    return {"ok": True, "filename": safe_name, "path": dest, "source": "uploaded"}
