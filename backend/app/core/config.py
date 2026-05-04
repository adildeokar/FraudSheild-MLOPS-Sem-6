from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path
import os


# backend/app/core/config.py → backend is 3 parents up
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# Repository root (parent of backend/) — full Kaggle creditcard.csv lives here
REPO_ROOT = BASE_DIR.parent


def resolve_primary_creditcard_csv() -> str:
    """
    Prefer explicit env, then repo-root creditcard.csv, then backend/data/creditcard.csv.
    Used for both default training and 'real' source when the primary file exists.
    """
    env = os.environ.get("CREDITCARD_DATA_FILE", "").strip()
    if env:
        ep = Path(env).expanduser()
        if ep.is_file() and ep.stat().st_size > 0:
            return str(ep.resolve())
    root_cc = REPO_ROOT / "creditcard.csv"
    if root_cc.is_file() and root_cc.stat().st_size > 0:
        return str(root_cc.resolve())
    return str((BASE_DIR / "data" / "creditcard.csv").resolve())


class Settings(BaseSettings):
    APP_NAME: str = "Credit Card Fraud Detection API"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/data/fraud_detection.db"

    DATA_DIR: str = str(BASE_DIR / "data")
    MODELS_DIR: str = str(BASE_DIR / "models")

    # Canonical fraud CSV (repo root creditcard.csv when present)
    TRAINING_DATA_FILE: str = Field(default_factory=resolve_primary_creditcard_csv)
    REAL_CREDITCARD_FILE: str = Field(default_factory=resolve_primary_creditcard_csv)

    # OpenML / prepare-real writes here only — never overwrites repo-root creditcard.csv
    REAL_OPENML_SAMPLE_PATH: str = str(BASE_DIR / "data" / "real_creditcard.csv")

    NEW_DATA_FILE: str = str(BASE_DIR / "data" / "New_transactions.csv")

    FRAUD_THRESHOLD: float = 0.5
    MODEL_IMPROVEMENT_THRESHOLD: float = 0.001

    # If true, startup builds REAL_OPENML_SAMPLE_PATH from OpenML when primary CSV is missing
    AUTO_PREPARE_REAL_DATASET: bool = True
    # Stratified cap for real CSV (smaller = faster training, still real).
    REAL_DATA_MAX_ROWS: int = 28000

    CORS_ORIGINS: list = ["http://localhost:3000", "http://frontend:3000", "*"]

    class Config:
        env_file = ".env"


settings = Settings()

os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.MODELS_DIR, exist_ok=True)
