"""
Auto-Retraining Pipeline — Credit Card Fraud Detection
The CORE INNOVATION: performance-gated model update.
"""

import json
import joblib
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from sklearn.model_selection import train_test_split

from app.core.config import settings
from app.ml.pipeline import (
    load_data, preprocess, apply_smote, get_models,
    evaluate_model, get_feature_importance, get_next_version,
    set_production_model, load_production_model, FEATURE_COLS
)
from datetime import datetime


def retrain_with_new_data(new_data_path: str, db=None) -> Dict[str, Any]:
    """
    Retraining pipeline:
    1. Load new data + existing data
    2. Train new model suite
    3. Compare with current production model
    4. Upgrade ONLY if performance improves
    5. Log full comparison
    """
    if not os.path.exists(new_data_path):
        raise FileNotFoundError(f"New data file not found: {new_data_path}")

    print(f"[Retrain] Loading new data from {new_data_path}")
    df_new = load_data(new_data_path)

    # Load existing training data to combine
    existing_path = settings.TRAINING_DATA_FILE
    if os.path.exists(existing_path):
        print(f"[Retrain] Loading existing data from {existing_path}")
        df_existing = load_data(existing_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates().reset_index(drop=True)
        print(f"[Retrain] Combined dataset: {len(df_combined)} records")
    else:
        df_combined = df_new
        print(f"[Retrain] Using only new data: {len(df_combined)} records")

    X = df_combined[FEATURE_COLS]
    y = df_combined["Class"].values
    dataset_size = len(df_combined)
    fraud_count = int(y.sum())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_scaled, scaler = preprocess(X_train, fit=True)
    X_test_scaled, _ = preprocess(X_test, scaler=scaler, fit=False)

    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    X_train_res, y_train_res = apply_smote(X_train_scaled.values, y_train)

    print(f"[Retrain] After SMOTE — train size: {len(X_train_res)}")

    models = get_models(scale_pos_weight=scale_pos)
    results = {}

    for name, model in models.items():
        print(f"[Retrain] Training {name}...")
        model.fit(X_train_res, y_train_res)
        metrics = evaluate_model(model, X_test_scaled.values, y_test)
        results[name] = {"model": model, "metrics": metrics}
        print(f"  {name}: ROC-AUC={metrics['roc_auc']:.4f} | F1={metrics['f1_score']:.4f}")

    best_name = max(results, key=lambda n: results[n]["metrics"]["roc_auc"])
    best_model = results[best_name]["model"]
    best_metrics = results[best_name]["metrics"]
    new_roc_auc = best_metrics["roc_auc"]

    # --- Get current production model metrics ---
    prod_file = Path(settings.MODELS_DIR) / "production.json"
    current_roc_auc = 0.0
    current_version = "none"
    current_model_type = "none"

    if prod_file.exists():
        with open(prod_file) as f:
            prod_info = json.load(f)
        current_roc_auc = prod_info.get("metrics", {}).get("roc_auc", 0.0)
        current_version = prod_info.get("version", "none")
        current_model_type = prod_info.get("model_type", "none")

    improvement = new_roc_auc - current_roc_auc
    improved = improvement > settings.MODEL_IMPROVEMENT_THRESHOLD

    print(f"\n[Retrain] === MODEL COMPARISON ===")
    print(f"  Current production: {current_version} ({current_model_type}) — ROC-AUC={current_roc_auc:.4f}")
    print(f"  New candidate:      {best_name} — ROC-AUC={new_roc_auc:.4f}")
    print(f"  Improvement:        {improvement:+.4f}")
    print(f"  Decision:           {'✅ UPGRADE' if improved else '❌ KEEP CURRENT'}")

    version = get_next_version()
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_path = str(Path(settings.MODELS_DIR) / f"model_{version}_{timestamp}.joblib")
    scaler_path = str(Path(settings.MODELS_DIR) / f"scaler_{version}_{timestamp}.joblib")

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    comparison_note = (
        f"Retrain comparison: {current_version}(ROC={current_roc_auc:.4f}) vs "
        f"{version}(ROC={new_roc_auc:.4f}). Improvement={improvement:+.4f}. "
        f"Decision: {'UPGRADED' if improved else 'KEPT OLD'}. "
        f"All models: {json.dumps({n: r['metrics']['roc_auc'] for n, r in results.items()})}"
    )

    if improved:
        set_production_model(version, model_path, scaler_path, best_name, best_metrics)
        print(f"[Retrain] Production model upgraded to {version}")
    else:
        print(f"[Retrain] New model saved as {version} but NOT promoted to production")

    all_model_metrics = {name: results[name]["metrics"] for name in results}
    feature_importance = get_feature_importance(best_model, best_name)

    if db is not None:
        try:
            from app.db.models import ModelRegistry
            if improved:
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
                is_production=improved,
                comparison_notes=comparison_note
            )
            db.add(reg)
            db.commit()
        except Exception as e:
            print(f"[Retrain] DB error: {e}")

    return {
        "status": "success",
        "improved": improved,
        "improvement": round(improvement, 6),
        "old_model": {
            "version": current_version,
            "model_type": current_model_type,
            "roc_auc": round(current_roc_auc, 6),
        },
        "new_model": {
            "version": version,
            "model_type": best_name,
            "roc_auc": round(new_roc_auc, 6),
            "metrics": best_metrics,
        },
        "all_model_metrics": all_model_metrics,
        "feature_importance": feature_importance,
        "dataset_size": dataset_size,
        "fraud_count": fraud_count,
        "message": (
            f"Production model upgraded from {current_version} to {version} "
            f"(ROC-AUC improved by {improvement:+.4f})"
            if improved else
            f"New model {version} trained but not promoted. "
            f"Current model {current_version} remains in production "
            f"(improvement {improvement:+.4f} below threshold {settings.MODEL_IMPROVEMENT_THRESHOLD})"
        ),
    }
