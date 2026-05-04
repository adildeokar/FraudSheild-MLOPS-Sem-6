"""
Active training dataset source (synthetic, real, or uploaded).
Persists to data/data_source.json — does not break existing defaults when missing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

from app.core.config import settings

DataSourceKind = Literal["synthetic", "real", "uploaded"]


def _state_path() -> Path:
    return Path(settings.DATA_DIR) / "data_source.json"


def _default_state() -> dict:
    return {
        "source": "synthetic",
        "upload_filename": "uploaded_custom.csv",
    }


def read_state() -> dict:
    p = _state_path()
    if not p.exists():
        return _default_state()
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        if "source" not in data:
            data["source"] = "synthetic"
        if "upload_filename" not in data:
            data["upload_filename"] = "uploaded_custom.csv"
        return data
    except Exception:
        return _default_state()


def write_state(source: DataSourceKind, upload_filename: str | None = None) -> dict:
    st = read_state()
    st["source"] = source
    if upload_filename:
        st["upload_filename"] = upload_filename
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    with open(_state_path(), "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)
    return st


def get_active_training_data_path() -> str:
    """Resolved CSV used by training and retraining (existing base data)."""
    st = read_state()
    src = st.get("source", "synthetic")

    if src == "real":
        p = Path(settings.REAL_CREDITCARD_FILE)
        if p.exists() and p.stat().st_size > 0:
            return str(p)
        # Optional smaller OpenML sample when primary CSV is absent
        sample = Path(settings.REAL_OPENML_SAMPLE_PATH)
        if sample.exists() and sample.stat().st_size > 0:
            return str(sample)
        return settings.TRAINING_DATA_FILE

    if src == "uploaded":
        name = st.get("upload_filename") or "uploaded_custom.csv"
        p = Path(settings.DATA_DIR) / name
        if p.exists() and p.stat().st_size > 0:
            return str(p)
        return settings.TRAINING_DATA_FILE

    return settings.TRAINING_DATA_FILE


def get_source_label() -> str:
    """Configured dataset label for APIs (synthetic | real | uploaded)."""
    return str(read_state().get("source", "synthetic"))


def get_upload_path_for_state() -> Path:
    st = read_state()
    name = st.get("upload_filename") or "uploaded_custom.csv"
    return Path(settings.DATA_DIR) / name
