"""
FastAPI Application — Credit Card Fraud Detection System
Startup: generates data, trains model, initializes DB.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import sys

from app.core.config import settings
from app.db.database import init_db

# ---------------------------------------------------------------------------
# Global model cache (hot-swappable in-memory model)
# ---------------------------------------------------------------------------
model_cache = {
    "model": None,
    "scaler": None,
    "version": None,
    "model_type": None,
}


def reload_model_cache():
    """Load (or reload) the production model into memory."""
    global model_cache
    try:
        from app.ml.pipeline import load_production_model
        model, scaler, info = load_production_model()
        model_cache["model"] = model
        model_cache["scaler"] = scaler
        model_cache["version"] = info.get("version", "unknown")
        model_cache["model_type"] = info.get("model_type", "unknown")
        print(f"[Cache] Loaded model {model_cache['version']} ({model_cache['model_type']})")
    except FileNotFoundError:
        print("[Cache] No production model found yet.")
    except Exception as e:
        print(f"[Cache] Error loading model: {e}")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description=(
        "Credit Card Fraud Detection MLOps System. "
        "Real-time fraud prediction with auto-retraining pipeline."
    ),
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------
from app.api.routes import predict, retrain, metrics, data_ops, insights, operations

app.include_router(predict.router, prefix="/api")
app.include_router(retrain.router, prefix="/api")
app.include_router(metrics.router, prefix="/api")
app.include_router(data_ops.router, prefix="/api")
app.include_router(insights.router, prefix="/api")
app.include_router(operations.router, prefix="/api")


# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print("  Credit Card Fraud Detection - MLOps System")
    print("=" * 60)

    # Initialize database
    print("[Startup] Initializing database...")
    init_db()

    # Generate synthetic data if not present
    if not os.path.exists(settings.TRAINING_DATA_FILE):
        print("[Startup] Generating synthetic training data...")
        # Generator always writes under backend/data (even if primary CSV is repo-root)
        data_dir = settings.DATA_DIR
        sys.path.insert(0, data_dir)
        try:
            from data.generate_data import main as gen_data
            gen_data()
        except ImportError:
            # Try direct execution
            import subprocess
            gen_script = os.path.join(data_dir, "generate_data.py")
            if os.path.exists(gen_script):
                subprocess.run([sys.executable, gen_script], check=True)
            else:
                print("[Startup] Warning: Could not find data generator script")
    else:
        print(f"[Startup] Training data found: {settings.TRAINING_DATA_FILE}")

    # OpenML sample only when no primary training CSV exists (never overwrites repo creditcard.csv)
    from pathlib import Path

    primary_train = Path(settings.TRAINING_DATA_FILE)
    openml_sample = Path(settings.REAL_OPENML_SAMPLE_PATH)
    if (
        settings.AUTO_PREPARE_REAL_DATASET
        and not primary_train.exists()
        and not openml_sample.exists()
    ):
        print("[Startup] Building real_creditcard.csv from OpenML (first run may take 1-2 min)...")
        try:
            from data.real_data_loader import build_real_creditcard_file
            from app.core.data_source import write_state

            p, n, fc = build_real_creditcard_file(out_path=str(openml_sample))
            write_state("real")
            print(f"[Startup] Real training CSV ready: {p} ({n} rows, fraud={fc})")
        except Exception as ex:
            print(f"[Startup] Real dataset fetch skipped: {ex}")

    # Generate New_transactions.csv if not present
    if not os.path.exists(settings.NEW_DATA_FILE):
        print("[Startup] Generating new transactions data...")
        try:
            from data.generate_data import generate_new_transactions
            import pandas as pd
            df_new = generate_new_transactions()
            df_new.to_csv(settings.NEW_DATA_FILE, index=False)
            print(f"[Startup] New transactions saved: {len(df_new)} records")
        except Exception as e:
            print(f"[Startup] Warning: Could not generate new transactions: {e}")

    # One-time refresh if production was fit on unrealistically perfect (ROC ~ 1) toy data
    prod_file = Path(settings.MODELS_DIR) / "production.json"
    realism_flag = Path(settings.DATA_DIR) / ".metrics_realism_refresh_v1"
    if (
        prod_file.exists()
        and not realism_flag.exists()
        and os.environ.get("FRAUDSHIELD_SKIP_METRIC_REFRESH", "").lower() not in ("1", "true", "yes")
    ):
        try:
            with open(prod_file, encoding="utf-8") as f:
                pj = json.load(f)
            roc = float(pj.get("metrics", {}).get("roc_auc", 0))
            if roc >= 0.999:
                print("[Startup] ROC-AUC ~1.0 detected - refitting on real (or refreshed synthetic) data once...")
                from app.core.data_source import get_active_training_data_path, write_state

                primary_train = Path(settings.TRAINING_DATA_FILE)
                openml_sample = Path(settings.REAL_OPENML_SAMPLE_PATH)
                if primary_train.exists():
                    write_state("synthetic")
                elif not openml_sample.exists():
                    if settings.AUTO_PREPARE_REAL_DATASET:
                        try:
                            from data.real_data_loader import build_real_creditcard_file

                            build_real_creditcard_file(out_path=str(openml_sample))
                            write_state("real")
                            print("[Startup] real_creditcard.csv created for realism pass.")
                        except Exception as ex:
                            print(f"[Startup] Real fetch failed ({ex}); regenerating overlapping synthetic CSV...")
                            data_dir = settings.DATA_DIR
                            sys.path.insert(0, data_dir)
                            from data.generate_data import main as gen_data

                            gen_data()
                            write_state("synthetic")
                    else:
                        print("[Startup] Regenerating synthetic training data (overlapping classes)...")
                        data_dir = settings.DATA_DIR
                        sys.path.insert(0, data_dir)
                        from data.generate_data import main as gen_data

                        gen_data()
                        write_state("synthetic")
                else:
                    write_state("real")

                train_csv = get_active_training_data_path()
                from app.db.database import SessionLocal
                from app.ml.pipeline import run_training_pipeline

                _db = SessionLocal()
                try:
                    run_training_pipeline(train_csv, db=_db)
                finally:
                    _db.close()
                realism_flag.write_text("ok", encoding="utf-8")
                print("[Startup] Model refit complete - dashboard metrics should look realistic.")
        except Exception as ex:
            print(f"[Startup] Realism refresh skipped: {ex}")

    # Train initial model if none exists
    if not prod_file.exists():
        print("[Startup] No production model found. Training initial model...")
        from app.core.data_source import get_active_training_data_path
        train_csv = get_active_training_data_path()
        if os.path.exists(train_csv):
            try:
                from app.db.database import SessionLocal
                from app.ml.pipeline import run_training_pipeline
                db = SessionLocal()
                result = run_training_pipeline(train_csv, db=db)
                db.close()
                print(f"[Startup] Initial model trained: {result['version']} ({result['model_type']})")
                print(f"[Startup] ROC-AUC: {result['metrics']['roc_auc']:.4f}")
            except Exception as e:
                print(f"[Startup] Training error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("[Startup] No training data available. Skipping model training.")
    else:
        print("[Startup] Production model found.")

    # Load model into cache
    reload_model_cache()

    print("[Startup] System ready!")
    print(f"[Startup] API docs: http://localhost:8000/docs")
    print("=" * 60)


@app.get("/", tags=["System"])
async def root():
    return {
        "name": settings.APP_NAME,
        "version": settings.VERSION,
        "status": "running",
        "model_loaded": model_cache["model"] is not None,
        "endpoints": {
            "predict": "/api/predict",
            "retrain": "/api/retrain",
            "metrics": "/api/metrics",
            "stats": "/api/stats",
            "history": "/api/history",
            "health": "/api/health",
            "data_source": "/api/data/source",
            "drift": "/api/drift",
            "explain": "/api/explain",
            "stream": "/api/stream/status",
            "docs": "/docs"
        }
    }
