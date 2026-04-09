# 📄 SYSTEM EXPLANATION — Credit Card Fraud Detection MLOps System
## (Viva Voce Script & Technical Documentation)

---

## 1. PROBLEM DEFINITION

Credit card fraud is a critical problem in the financial sector. Every year, billions of dollars are lost to fraudulent transactions. The challenge is detecting these fraudulent transactions in **real-time** from among millions of legitimate ones.

**Key Challenges:**
- **Class Imbalance**: Fraud transactions are extremely rare (~0.17% of all transactions), making it hard for models to learn fraud patterns
- **Real-time Requirement**: Decisions must be made in milliseconds to avoid blocking legitimate transactions
- **Concept Drift**: Fraud patterns evolve over time, requiring models to be continuously updated
- **High Stakes**: False negatives (missed fraud) cause financial loss; false positives (wrong alerts) cause customer dissatisfaction

---

## 2. OBJECTIVE

Build a **complete MLOps pipeline** that:
1. Trains a machine learning model to detect credit card fraud
2. Serves real-time predictions via a REST API
3. Automatically retrains when new transaction data arrives
4. Compares new vs old model performance and only upgrades if the new model is better
5. Provides a professional dashboard to monitor all system activity

---

## 3. ARCHITECTURE

### Layer 1: Data Layer
- **creditcard.csv**: Primary training dataset with 30 features (Time, V1–V28, Amount, Class)
- **New_transactions.csv**: Simulated new data that arrives periodically, triggering retraining
- **SQLite Database**: Stores all predictions, transactions, and model metadata

### Layer 2: ML Pipeline Layer
- **Preprocessing**: StandardScaler for normalization, SMOTE for handling class imbalance
- **Training**: Three models trained simultaneously — Logistic Regression, Random Forest, XGBoost
- **Evaluation**: Full metrics calculated — Accuracy, Precision, Recall, F1, ROC-AUC
- **Model Selection**: Best model selected based on ROC-AUC score
- **Versioning**: Each model saved with version number and timestamp

### Layer 3: API Layer (FastAPI)
- **POST /predict**: Accept transaction features, return fraud prediction with probability
- **POST /retrain**: Trigger the retraining pipeline
- **GET /metrics**: Return current model performance metrics
- **GET /stats**: Return fraud statistics (total predictions, fraud rate, etc.)
- **GET /history**: Return prediction history
- **GET /model-registry**: Return all model versions and their performance
- **GET /health**: System health check

### Layer 4: UI Layer (Next.js + React)
- **Dashboard**: Real-time fraud statistics and KPIs
- **Live Prediction**: Form to submit transactions and see fraud predictions
- **Analytics**: Charts showing fraud trends and model performance
- **Retraining Panel**: Trigger retraining and see before/after comparison

---

## 4. MODULES EXPLAINED

### Module 1: Data Generation (`generate_data.py`)
Generates synthetic credit card transaction data that mimics real-world distributions:
- Normal transactions: Log-normal amount distribution, Gaussian V-features
- Fraud transactions: Different statistical properties, small amounts or large spikes
- Saves `creditcard.csv` (15,000 records) and `New_transactions.csv` (3,000 records)

### Module 2: ML Pipeline (`pipeline.py`)
The core machine learning engine:
```
Input Data → Validation → Split (80/20) → Scale (StandardScaler)
→ SMOTE (training only) → Train 3 Models → Evaluate → Select Best
→ Save Model + Scaler → Register in DB
```

### Module 3: Prediction Service (`predict.py`)
Handles real-time fraud detection:
```
API Request → Validate Features → Load Scaler → Transform
→ Model.predict_proba() → Threshold (0.5) → Risk Level → Store → Respond
```
Risk Levels:
- LOW: probability < 0.3
- MEDIUM: 0.3 ≤ probability < 0.6
- HIGH: 0.6 ≤ probability < 0.85
- CRITICAL: probability ≥ 0.85

### Module 4: Auto-Retraining System (`retrain.py`) ★ INNOVATION
```
Trigger → Load New_transactions.csv → Merge with creditcard.csv
→ Full Training Pipeline → New Model Metrics → Compare with Production
→ IF new_roc_auc > current_roc_auc + 0.001: SWAP MODEL
→ Log comparison → Update Dashboard
```
This ensures the production model **only ever gets better**.

### Module 5: Model Registry
SQLite table tracking:
- Model version
- Algorithm used (best model)
- All performance metrics
- Is it the current production model
- Training timestamp
- Dataset size used

---

## 5. WORKFLOW

### Training Workflow:
1. Load `creditcard.csv`
2. Validate schema and data quality
3. Split into 80% train, 20% test (stratified by class)
4. Apply StandardScaler on Amount and Time features
5. Apply SMOTE on training data to balance classes
6. Train Logistic Regression, Random Forest, and XGBoost
7. Evaluate all three on the same test set
8. Select the model with highest ROC-AUC
9. Save model and scaler with version number
10. Register in model_registry table

### Prediction Workflow:
1. Receive POST /predict request with transaction features
2. Validate all 30 features are present and valid
3. Load current production scaler
4. Scale Amount and Time features
5. Run inference on current production model
6. Get fraud probability from predict_proba()
7. Apply threshold (0.5) for binary classification
8. Determine risk level based on probability
9. Store prediction in predictions table
10. Return JSON response with prediction, probability, risk level

### Retraining Workflow:
1. Trigger via POST /retrain (manual or automated)
2. Load New_transactions.csv
3. Merge with existing training data
4. Run complete training pipeline on combined data
5. Evaluate new model on held-out test set
6. Compare: new ROC-AUC vs current production ROC-AUC
7. If new model is better (by > 0.001): save as production, update DB flag
8. If not better: save for reference but keep old production model
9. Log full comparison results
10. Return comparison metrics to frontend

---

## 6. TECHNOLOGY STACK

| Component | Technology | Purpose |
|-----------|-----------|---------|
| ML Framework | scikit-learn | LR, RF, preprocessing |
| Advanced ML | XGBoost | Gradient boosting classifier |
| Imbalance | imbalanced-learn | SMOTE oversampling |
| Serialization | joblib | Model save/load |
| API | FastAPI | REST API with async support |
| Validation | Pydantic | Request/response schemas |
| Database | SQLite + SQLAlchemy | Data persistence |
| Frontend | Next.js 14 | React framework with SSR |
| Styling | Tailwind CSS | Utility-first CSS |
| Charts | Recharts | Data visualization |
| Container | Docker | Application packaging |
| Orchestration | Docker Compose | Multi-service deployment |

---

## 7. INNOVATION

### Innovation 1: Performance-Gated Auto-Retraining ★
Unlike simple scheduled retraining, our system evaluates the new model before deploying it. The production model is only replaced if the new model demonstrates measurable improvement (ROC-AUC improvement > 0.001). This prevents degradation from poor-quality new data.

### Innovation 2: Multi-Model Competition
Three models compete on every training run. The best one wins. This ensures we always use the most appropriate algorithm for the current data distribution.

### Innovation 3: Zero-Downtime Model Swap
When a better model is found, it's hot-swapped into the prediction service without restarting the API. The model is loaded into memory from disk and the in-memory reference is updated atomically.

### Innovation 4: Feature Importance Dashboard
The system displays which transaction features (V1–V28, Amount, Time) most influence fraud decisions, giving interpretability to what is otherwise a "black box" prediction.

### Innovation 5: Complete Audit Trail
Every prediction, every retraining event, and every model version is logged to the database. This creates a full audit trail for regulatory compliance.

---

## 8. EXPECTED OUTPUT

### API Response (Normal Transaction):
```json
{
  "transaction_id": "a3f7c1d2-...",
  "is_fraud": false,
  "fraud_probability": 0.023,
  "risk_level": "LOW",
  "model_version": "v2",
  "model_type": "RandomForest",
  "processing_time_ms": 8,
  "timestamp": "2024-01-15T10:30:00"
}
```

### API Response (Fraud Transaction):
```json
{
  "transaction_id": "b9e2a4c5-...",
  "is_fraud": true,
  "fraud_probability": 0.923,
  "risk_level": "CRITICAL",
  "model_version": "v2",
  "model_type": "RandomForest",
  "processing_time_ms": 9,
  "timestamp": "2024-01-15T10:31:00"
}
```

### Retraining Response:
```json
{
  "status": "success",
  "old_model": { "version": "v1", "roc_auc": 0.951, "model_type": "RandomForest" },
  "new_model": { "version": "v2", "roc_auc": 0.967, "model_type": "XGBoost" },
  "improved": true,
  "improvement": 0.016,
  "message": "Production model upgraded from v1 to v2"
}
```

---

## 9. DEPLOYMENT

### Running with Docker (Recommended):
```bash
docker-compose up --build
```
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Running Locally (Development):
```bash
# Backend
cd backend && pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev
```

---

## 10. KEY TALKING POINTS FOR VIVA

1. **"Why SMOTE?"** — Because standard resampling would just duplicate fraud samples, causing overfitting. SMOTE generates synthetic fraud samples between real ones, making the model more robust.

2. **"Why three models?"** — Different algorithms capture different patterns. Logistic Regression provides a linear baseline, Random Forest captures non-linear interactions, XGBoost handles complex patterns with boosting.

3. **"Why ROC-AUC for model selection?"** — Accuracy is misleading for imbalanced datasets (a model saying everything is normal gets 99.83% accuracy). ROC-AUC measures the model's ability to distinguish between classes regardless of threshold.

4. **"Why the performance gate?"** — Without it, retraining on noisy or incomplete data could degrade the production model. The gate ensures we only ever improve.

5. **"What's novel?"** — The combination of automated competitive model selection + performance-gated deployment + real-time monitoring dashboard + zero-downtime hot-swap is what makes this a production-grade MLOps system, not just a model.
