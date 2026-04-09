from pydantic_settings import BaseSettings
from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    APP_NAME: str = "Credit Card Fraud Detection API"
    VERSION: str = "1.0.0"
    DEBUG: bool = True

    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/data/fraud_detection.db"

    DATA_DIR: str = str(BASE_DIR / "data")
    MODELS_DIR: str = str(BASE_DIR / "models")
    TRAINING_DATA_FILE: str = str(BASE_DIR / "data" / "creditcard.csv")
    NEW_DATA_FILE: str = str(BASE_DIR / "data" / "New_transactions.csv")

    FRAUD_THRESHOLD: float = 0.5
    MODEL_IMPROVEMENT_THRESHOLD: float = 0.001

    CORS_ORIGINS: list = ["http://localhost:3000", "http://frontend:3000", "*"]

    class Config:
        env_file = ".env"


settings = Settings()

os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.MODELS_DIR, exist_ok=True)
