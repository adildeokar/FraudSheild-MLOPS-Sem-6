"""
FastAPI Application — Credit Card Fraud Detection System
Startup: generates data, trains model, initializes DB.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
from app.api.routes import predict, retrain, metrics

app.include_router(predict.router, prefix="/api")
app.include_router(retrain.router, prefix="/api")
app.include_router(metrics.router, prefix="/api")


# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    print("=" * 60)
    print("  Credit Card Fraud Detection — MLOps System")
    print("=" * 60)

    # Initialize database
    print("[Startup] Initializing database...")
    init_db()

    # Generate synthetic data if not present
    if not os.path.exists(settings.TRAINING_DATA_FILE):
        print("[Startup] Generating synthetic training data...")
        # Add data directory to path for the generator
        data_dir = os.path.dirname(settings.TRAINING_DATA_FILE)
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

    # Train initial model if none exists
    from pathlib import Path
    prod_file = Path(settings.MODELS_DIR) / "production.json"
    if not prod_file.exists():
        print("[Startup] No production model found. Training initial model...")
        if os.path.exists(settings.TRAINING_DATA_FILE):
            try:
                from app.db.database import SessionLocal
                from app.ml.pipeline import run_training_pipeline
                db = SessionLocal()
                result = run_training_pipeline(settings.TRAINING_DATA_FILE, db=db)
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
            "docs": "/docs"
        }
    }
