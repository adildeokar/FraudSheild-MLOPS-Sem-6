"""
ML Pipeline — Credit Card Fraud Detection
Handles preprocessing, training, evaluation, and model persistence.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from app.core.config import settings


FEATURE_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
SCALE_COLS = ["Time", "Amount"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """Load and validate transaction CSV data."""
    df = pd.read_csv(filepath)
    required = set(FEATURE_COLS + ["Class"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    df = df.dropna()
    return df


def get_next_version() -> str:
    """Determine the next model version string (v1, v2, ...)."""
    models_dir = Path(settings.MODELS_DIR)
    existing = [f for f in models_dir.glob("model_v*.joblib")]
    if not existing:
        return "v1"
    versions = []
    for f in existing:
        try:
            num = int(f.stem.split("_")[1][1:])
            versions.append(num)
        except (IndexError, ValueError):
            pass
    return f"v{max(versions) + 1}" if versions else "v1"


def get_current_production_model_path() -> Optional[Tuple[str, str]]:
    """Return (model_path, scaler_path) for the latest production model, or None."""
    models_dir = Path(settings.MODELS_DIR)
    prod_file = models_dir / "production.json"
    if not prod_file.exists():
        return None
    with open(prod_file) as f:
        info = json.load(f)
    model_path = info.get("model_path")
    scaler_path = info.get("scaler_path")
    if model_path and scaler_path and os.path.exists(model_path) and os.path.exists(scaler_path):
        return model_path, scaler_path
    return None


def set_production_model(version: str, model_path: str, scaler_path: str, model_type: str, metrics: Dict):
    """Write the production.json pointer file."""
    prod_file = Path(settings.MODELS_DIR) / "production.json"
    info = {
        "version": version,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "model_type": model_type,
        "metrics": metrics,
        "set_at": datetime.utcnow().isoformat()
    }
    with open(prod_file, "w") as f:
        json.dump(info, f, indent=2)


def load_production_model() -> Tuple[Any, StandardScaler, Dict]:
    """Load the current production model, scaler, and metadata."""
    prod_file = Path(settings.MODELS_DIR) / "production.json"
    if not prod_file.exists():
        raise FileNotFoundError("No production model found. Please train first.")
    with open(prod_file) as f:
        info = json.load(f)
    model = joblib.load(info["model_path"])
    scaler = joblib.load(info["scaler_path"])
    return model, scaler, info


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(X: pd.DataFrame, scaler: Optional[StandardScaler] = None, fit: bool = True):
    """Scale Time and Amount; optionally fit scaler."""
    X = X.copy()
    if fit:
        scaler = StandardScaler()
        X[SCALE_COLS] = scaler.fit_transform(X[SCALE_COLS])
        return X, scaler
    else:
        X[SCALE_COLS] = scaler.transform(X[SCALE_COLS])
        return X, scaler


def apply_smote(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to oversample minority (fraud) class on training data only."""
    fraud_count = y.sum()
    if fraud_count < 5:
        return X, y
    k_neighbors = min(5, fraud_count - 1)
    sm = SMOTE(sampling_strategy=0.3, k_neighbors=k_neighbors, random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models(scale_pos_weight: float = 10.0) -> Dict[str, Any]:
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            max_depth=10
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0
        )
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Compute all evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 6),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 6),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def get_feature_importance(model, model_type: str) -> Dict[str, float]:
    """Extract feature importance from tree-based models."""
    feature_names = FEATURE_COLS
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            return {name: round(float(imp), 6) for name, imp in zip(feature_names, importances)}
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
            importances = importances / importances.sum()
            return {name: round(float(imp), 6) for name, imp in zip(feature_names, importances)}
    except Exception:
        pass
    return {name: 0.0 for name in feature_names}


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------

def run_training_pipeline(data_path: str, db=None) -> Dict[str, Any]:
    """
    Full training pipeline:
    Load → Preprocess → SMOTE → Train all models → Evaluate → Select best → Save
    Returns info about the trained model.
    """
    print(f"[Pipeline] Loading data from {data_path}")
    df = load_data(data_path)

    X = df[FEATURE_COLS]
    y = df["Class"].values
    dataset_size = len(df)
    fraud_count = int(y.sum())

    print(f"[Pipeline] Dataset: {dataset_size} records | Fraud: {fraud_count} ({fraud_count/dataset_size*100:.2f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train_scaled, scaler = preprocess(X_train, fit=True)
    X_test_scaled, _ = preprocess(X_test, scaler=scaler, fit=False)

    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    X_train_res, y_train_res = apply_smote(X_train_scaled.values, y_train)

    print(f"[Pipeline] After SMOTE — train size: {len(X_train_res)} | fraud: {y_train_res.sum()}")

    models = get_models(scale_pos_weight=scale_pos)
    results = {}

    for name, model in models.items():
        print(f"[Pipeline] Training {name}...")
        t0 = time.time()
        model.fit(X_train_res, y_train_res)
        elapsed = round((time.time() - t0) * 1000, 1)
        metrics = evaluate_model(model, X_test_scaled.values, y_test)
        metrics["training_time_ms"] = elapsed
        results[name] = {"model": model, "metrics": metrics}
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1_score']:.4f} | Time: {elapsed}ms")

    best_name = max(results, key=lambda n: results[n]["metrics"]["roc_auc"])
    best_model = results[best_name]["model"]
    best_metrics = results[best_name]["metrics"]
    print(f"[Pipeline] Best model: {best_name} (ROC-AUC={best_metrics['roc_auc']:.4f})")

    version = get_next_version()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_filename = f"model_{version}_{timestamp}.joblib"
    scaler_filename = f"scaler_{version}_{timestamp}.joblib"
    model_path = str(Path(settings.MODELS_DIR) / model_filename)
    scaler_path = str(Path(settings.MODELS_DIR) / scaler_filename)

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"[Pipeline] Saved model -> {model_path}")

    set_production_model(version, model_path, scaler_path, best_name, best_metrics)

    feature_importance = get_feature_importance(best_model, best_name)
    all_model_metrics = {name: results[name]["metrics"] for name in results}

    if db is not None:
        try:
            from app.db.models import ModelRegistry
            existing = db.query(ModelRegistry).filter(ModelRegistry.version == version).first()
            if not existing:
                db.query(ModelRegistry).update({ModelRegistry.is_production: False})
                reg = ModelRegistry(
                    version=version,
                    model_type=best_name,
                    model_path=model_path,
                    scaler_path=scaler_path,
                    accuracy=best_metrics["accuracy"],
                    precision=best_metrics["precision"],
                    recall=best_metrics["recall"],
                    f1_score=best_metrics["f1_score"],
                    roc_auc=best_metrics["roc_auc"],
                    true_negatives=best_metrics["true_negatives"],
                    false_positives=best_metrics["false_positives"],
                    false_negatives=best_metrics["false_negatives"],
                    true_positives=best_metrics["true_positives"],
                    dataset_size=dataset_size,
                    fraud_count=fraud_count,
                    is_production=True,
                    comparison_notes=f"Initial training. All models: {json.dumps(all_model_metrics)}"
                )
                db.add(reg)
                db.commit()
        except Exception as e:
            print(f"[Pipeline] DB registration error: {e}")

    return {
        "version": version,
        "model_type": best_name,
        "metrics": best_metrics,
        "all_model_metrics": all_model_metrics,
        "feature_importance": feature_importance,
        "dataset_size": dataset_size,
        "fraud_count": fraud_count,
        "model_path": model_path,
        "scaler_path": scaler_path,
    }
