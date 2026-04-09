from sqlalchemy import Column, Integer, Float, String, Boolean, DateTime, Text
from sqlalchemy.sql import func
from app.db.database import Base


class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, unique=True, index=True)
    time_feature = Column(Float)
    amount = Column(Float)
    v1 = Column(Float); v2 = Column(Float); v3 = Column(Float)
    v4 = Column(Float); v5 = Column(Float); v6 = Column(Float)
    v7 = Column(Float); v8 = Column(Float); v9 = Column(Float)
    v10 = Column(Float); v11 = Column(Float); v12 = Column(Float)
    v13 = Column(Float); v14 = Column(Float); v15 = Column(Float)
    v16 = Column(Float); v17 = Column(Float); v18 = Column(Float)
    v19 = Column(Float); v20 = Column(Float); v21 = Column(Float)
    v22 = Column(Float); v23 = Column(Float); v24 = Column(Float)
    v25 = Column(Float); v26 = Column(Float); v27 = Column(Float)
    v28 = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(String, index=True)
    is_fraud = Column(Boolean)
    fraud_probability = Column(Float)
    risk_level = Column(String)
    model_version = Column(String)
    model_type = Column(String)
    processing_time_ms = Column(Float)
    amount = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, unique=True, index=True)
    model_type = Column(String)
    model_path = Column(String)
    scaler_path = Column(String)

    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    roc_auc = Column(Float)

    true_negatives = Column(Integer)
    false_positives = Column(Integer)
    false_negatives = Column(Integer)
    true_positives = Column(Integer)

    dataset_size = Column(Integer)
    fraud_count = Column(Integer)
    is_production = Column(Boolean, default=False)

    comparison_notes = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
