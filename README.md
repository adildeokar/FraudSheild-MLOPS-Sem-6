# 🛡️ FraudShield — Credit Card Fraud Detection MLOps System

A complete, production-grade MLOps pipeline for real-time credit card fraud detection with auto-retraining.

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
docker-compose up --build
```
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

> First run takes ~2 minutes to train the initial model.

---

### Option 2: Local Development

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

---

## 📁 Project Structure

```
MLOPS JURY/
├── MINDMAP.md                    # Complete system mind map
├── SYSTEM_EXPLANATION.md         # Viva script & technical docs
├── docker-compose.yml            # One-command deployment
│
├── backend/
│   ├── app/
│   │   ├── main.py               # FastAPI app + startup init
│   │   ├── core/config.py        # Settings
│   │   ├── db/                   # SQLAlchemy models + database
│   │   ├── ml/
│   │   │   ├── pipeline.py       # Training pipeline
│   │   │   └── retrain.py        # Auto-retraining (CORE INNOVATION)
│   │   └── api/routes/
│   │       ├── predict.py        # /predict endpoint
│   │       ├── retrain.py        # /retrain endpoint
│   │       └── metrics.py        # /metrics, /stats endpoints
│   ├── data/
│   │   ├── generate_data.py      # Synthetic data generator
│   │   ├── creditcard.csv        # (generated on first run)
│   │   └── New_transactions.csv  # (generated on first run)
│   └── models/                   # Saved model files (auto-created)
│
└── frontend/
    └── src/app/
        ├── page.tsx              # Dashboard
        ├── predict/page.tsx      # Live Prediction
        ├── analytics/page.tsx    # Analytics & Charts
        └── retrain/page.tsx      # Retraining Panel
```

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Real-time Prediction** | Sub-50ms fraud detection via ML model |
| **3-Model Competition** | LR vs Random Forest vs XGBoost |
| **Auto-Retraining** | Trigger retraining with new data |
| **Performance Gate** | Production updated ONLY if ROC-AUC improves |
| **Model Registry** | Full version history in SQLite |
| **Feature Importance** | Visual explainability of fraud factors |
| **Fraud Analytics** | Charts, trends, risk distribution |
| **Audit Trail** | Every prediction logged to database |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Predict if transaction is fraudulent |
| POST | `/api/retrain` | Trigger auto-retraining pipeline |
| GET | `/api/metrics` | Current model performance metrics |
| GET | `/api/stats` | Fraud statistics & daily trends |
| GET | `/api/history` | Prediction history |
| GET | `/api/model-registry` | All model versions |
| GET | `/api/feature-importance` | Feature importance scores |
| GET | `/api/health` | System health check |

---

## 🧠 ML Pipeline

```
Data (CSV) → Validation → Train/Test Split (80/20)
→ StandardScaler → SMOTE → Train 3 Models
→ Evaluate (Accuracy, Precision, Recall, F1, ROC-AUC)
→ Select Best (by ROC-AUC) → Save with Versioning
→ Deploy to Production
```

### Auto-Retraining (Innovation):
```
New Data → Merge → Retrain → Compare ROC-AUC
→ IF new > old + 0.001: UPGRADE PRODUCTION
→ ELSE: Keep current model
```

---

## 📊 Tech Stack

- **ML**: scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **API**: FastAPI + Uvicorn
- **DB**: SQLite + SQLAlchemy
- **Frontend**: Next.js 14 + TypeScript + Tailwind CSS
- **Charts**: Recharts
- **Deploy**: Docker + Docker Compose
