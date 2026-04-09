# 🧠 MIND MAP — Credit Card Fraud Detection MLOps System

```
CREDIT CARD FRAUD DETECTION — MLOps System
│
├── 1. PROBLEM UNDERSTANDING
│   ├── Domain: Financial Transaction Security
│   ├── Challenge: Detect fraudulent transactions in real-time
│   ├── Core Issues
│   │   ├── Class Imbalance (~0.17% fraud vs 99.83% normal)
│   │   ├── High-stakes decisions (false negatives = financial loss)
│   │   ├── Real-time prediction requirement (milliseconds)
│   │   └── Model drift over time (fraud patterns evolve)
│   └── Key Metrics
│       ├── Recall (minimize missed fraud)
│       ├── Precision (minimize false alarms)
│       ├── F1 Score (balance precision/recall)
│       └── ROC-AUC (overall discriminative ability)
│
├── 2. SYSTEM MODULES
│   │
│   ├── 2.1 DATA INGESTION
│   │   ├── Source: CSV files (creditcard.csv, New_transactions.csv)
│   │   ├── Features: Time, V1–V28 (PCA), Amount, Class
│   │   ├── Validation: Schema check, null check, type enforcement
│   │   └── Storage: SQLite (transactions table)
│   │
│   ├── 2.2 DATA PREPROCESSING
│   │   ├── Feature Scaling
│   │   │   ├── StandardScaler on Amount + Time
│   │   │   └── V1–V28 already PCA-normalized
│   │   ├── Class Imbalance Handling
│   │   │   ├── SMOTE (Synthetic Minority Oversampling Technique)
│   │   │   └── class_weight='balanced' fallback
│   │   └── Train/Test Split (80/20, stratified)
│   │
│   ├── 2.3 MODEL TRAINING
│   │   ├── Logistic Regression (baseline)
│   │   │   └── class_weight='balanced', max_iter=1000
│   │   ├── Random Forest (primary)
│   │   │   └── 100 estimators, class_weight='balanced'
│   │   └── XGBoost (advanced)
│   │       └── scale_pos_weight, learning_rate=0.1
│   │
│   ├── 2.4 MODEL EVALUATION
│   │   ├── Accuracy
│   │   ├── Precision
│   │   ├── Recall
│   │   ├── F1 Score
│   │   ├── ROC-AUC
│   │   └── Confusion Matrix
│   │
│   ├── 2.5 MODEL SELECTION
│   │   ├── Primary criterion: ROC-AUC score
│   │   ├── All 3 models evaluated on same test set
│   │   └── Best model saved as production model
│   │
│   ├── 2.6 MODEL DEPLOYMENT
│   │   ├── Model versioning (v1, v2, v3...)
│   │   ├── Model registry in SQLite DB
│   │   ├── joblib serialization (.joblib files)
│   │   └── Hot-swap: replace production model without restart
│   │
│   ├── 2.7 PREDICTION SERVICE
│   │   ├── Input: 30 transaction features
│   │   ├── Preprocessing: scale Amount + Time
│   │   ├── Inference: fraud probability
│   │   ├── Output: prediction + probability + risk level
│   │   └── Logging: all predictions stored in DB
│   │
│   ├── 2.8 AUTO-RETRAINING PIPELINE ★ INNOVATION
│   │   ├── Trigger: New_transactions.csv detected / manual
│   │   ├── Process: Combine new + existing data
│   │   ├── Retrain: All 3 models on combined data
│   │   ├── Compare: New ROC-AUC vs Production ROC-AUC
│   │   ├── Gate: Update ONLY if improvement > threshold
│   │   └── Log: Full comparison stored in model_registry
│   │
│   └── 2.9 MONITORING & OBSERVABILITY
│       ├── Prediction history tracking
│       ├── Fraud rate over time
│       ├── Model performance trends
│       ├── Feature importance visualization
│       └── Model comparison dashboard
│
├── 3. COMPLETE WORKFLOW
│   │
│   ├── TRAINING FLOW
│   │   └── CSV Data → Load → Validate → Scale → SMOTE
│   │       → Train (LR + RF + XGB) → Evaluate → Select Best
│   │       → Save to models/ → Register in DB → Deploy
│   │
│   ├── PREDICTION FLOW
│   │   └── API Request → Validate Input → Load Scaler
│   │       → Preprocess → Model Inference → Threshold
│   │       → Risk Level → Store Prediction → Return Response
│   │
│   └── RETRAINING FLOW ★
│       └── Trigger → Load New Data → Merge with Old Data
│           → Retrain Pipeline → Evaluate New Model
│           → Compare with Production → If Better: Swap
│           → Log Results → Update Dashboard
│
├── 4. TECHNOLOGY STACK
│   │
│   ├── MACHINE LEARNING
│   │   ├── Python 3.11
│   │   ├── scikit-learn (LR, RF, preprocessing)
│   │   ├── XGBoost (gradient boosting)
│   │   ├── imbalanced-learn (SMOTE)
│   │   ├── pandas + numpy (data manipulation)
│   │   └── joblib (model serialization)
│   │
│   ├── BACKEND
│   │   ├── FastAPI (async REST API)
│   │   ├── SQLAlchemy + SQLite (ORM + DB)
│   │   ├── Pydantic (data validation)
│   │   └── Uvicorn (ASGI server)
│   │
│   ├── FRONTEND
│   │   ├── Next.js 14 (App Router)
│   │   ├── TypeScript
│   │   ├── Tailwind CSS (styling)
│   │   ├── Recharts (data visualization)
│   │   ├── Axios (API calls)
│   │   └── Lucide React (icons)
│   │
│   └── DEVOPS
│       ├── Docker (containerization)
│       ├── Docker Compose (orchestration)
│       └── GitHub (version control)
│
├── 5. INPUT → PROCESS → OUTPUT
│   │
│   ├── INPUT
│   │   ├── Transaction Features (Time, V1–V28, Amount)
│   │   ├── Historical Dataset (creditcard.csv)
│   │   └── New Transaction Batch (New_transactions.csv)
│   │
│   ├── PROCESS
│   │   ├── Feature Scaling (StandardScaler)
│   │   ├── Imbalance Correction (SMOTE)
│   │   ├── Multi-Model Training
│   │   ├── Automated Model Selection
│   │   ├── Real-time Inference
│   │   └── Performance-gated Model Update
│   │
│   └── OUTPUT
│       ├── Fraud Prediction (0 or 1)
│       ├── Fraud Probability (0.0 – 1.0)
│       ├── Risk Level (LOW / MEDIUM / HIGH / CRITICAL)
│       ├── Model Performance Metrics
│       ├── Feature Importance Rankings
│       └── Model Version History
│
└── 6. INNOVATION HIGHLIGHTS ★
    ├── Performance-gated Auto-retraining (only upgrades if better)
    ├── Multi-model comparison dashboard
    ├── Real-time fraud probability gauge
    ├── Feature importance visualization
    ├── Model version registry with history
    └── Hot-swappable production model (zero downtime)
```

---

## Architecture Diagram (ASCII)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CREDIT CARD FRAUD DETECTION SYSTEM               │
│                         MLOps Architecture                          │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────┐    ┌──────────────────────────────────────────────────┐
  │  React   │    │              FASTAPI BACKEND                      │
  │ Frontend │◄──►│  /predict  /retrain  /metrics  /health  /stats   │
  │ :3000    │    │                                                   │
  └──────────┘    │  ┌────────────┐  ┌────────────┐  ┌───────────┐  │
                  │  │ Prediction │  │ Retraining │  │  Metrics  │  │
                  │  │  Service   │  │  Pipeline  │  │  Service  │  │
                  │  └─────┬──────┘  └─────┬──────┘  └─────┬─────┘  │
                  │        │               │                │         │
                  │  ┌─────▼───────────────▼────────────────▼──────┐ │
                  │  │           ML PIPELINE LAYER                  │ │
                  │  │  ┌──────────┐ ┌──────────┐ ┌──────────────┐ │ │
                  │  │  │  Logistic│ │  Random  │ │   XGBoost    │ │ │
                  │  │  │Regression│ │  Forest  │ │   Classifier │ │ │
                  │  │  └──────────┘ └──────────┘ └──────────────┘ │ │
                  │  │           ↑ SMOTE + Scaler                   │ │
                  │  └──────────────────────────────────────────────┘ │
                  │                                                   │
                  │  ┌────────────────────┐  ┌─────────────────────┐ │
                  │  │    SQLite DB        │  │   Models Storage    │ │
                  │  │  - transactions    │  │  - model_v1.joblib  │ │
                  │  │  - predictions     │  │  - model_v2.joblib  │ │
                  │  │  - model_registry  │  │  - scaler_v1.joblib │ │
                  │  └────────────────────┘  └─────────────────────┘ │
                  └──────────────────────────────────────────────────┘
                                       │
                  ┌────────────────────┴───────────────────────────┐
                  │              DATA LAYER                         │
                  │  creditcard.csv        New_transactions.csv     │
                  │  (Training Data)       (Retraining Trigger)     │
                  └─────────────────────────────────────────────────┘
```
