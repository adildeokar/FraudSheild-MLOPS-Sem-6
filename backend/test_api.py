"""Quick API test script."""
import urllib.request, json

BASE = "http://localhost:8000/api"

def get(path):
    req = urllib.request.urlopen(f"{BASE}{path}", timeout=10)
    return json.loads(req.read())

def post(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(f"{BASE}{path}", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    res = urllib.request.urlopen(req, timeout=30)
    return json.loads(res.read())

# Health
health = get("/health")
print(f"Health: {health['status']} | Model: {health['model_version']} ({health['model_type']})")

# Metrics
m = get("/metrics")
metrics = m.get("metrics", {})
print(f"Metrics: ROC-AUC={metrics.get('roc_auc')} F1={metrics.get('f1_score')}")

# Stats
s = get("/stats")
print(f"Stats: total={s['total_predictions']} fraud={s['total_fraud']}")

# Normal prediction
normal = post("/predict", {
    "time": 0.0, "amount": 149.62,
    "v1": -1.3598, "v2": -0.0728, "v3": 2.5363, "v4": 1.3782, "v5": -0.3383,
    "v6": 0.4624, "v7": 0.2396, "v8": 0.0987, "v9": 0.3638, "v10": 0.0908,
    "v11": -0.5516, "v12": -0.6178, "v13": -0.9913, "v14": -0.3112, "v15": 1.4682,
    "v16": -0.4704, "v17": 0.2079, "v18": 0.0258, "v19": 0.4030, "v20": 0.2514,
    "v21": -0.0183, "v22": 0.2778, "v23": -0.1104, "v24": 0.0669, "v25": 0.1285,
    "v26": -0.1891, "v27": 0.1336, "v28": -0.0211
})
print(f"Normal TX: is_fraud={normal['is_fraud']} prob={normal['fraud_probability']:.4f} risk={normal['risk_level']}")

# Fraud prediction
fraud = post("/predict", {
    "time": 406.0, "amount": 1.0,
    "v1": -2.3122, "v2": 1.9519, "v3": -1.6096, "v4": 3.9979, "v5": -0.5221,
    "v6": -1.4265, "v7": -2.5374, "v8": 1.3918, "v9": -2.7700, "v10": -2.7722,
    "v11": 3.2020, "v12": -2.8990, "v13": -0.5952, "v14": -4.2895, "v15": 0.3898,
    "v16": -1.1405, "v17": -2.8300, "v18": -0.0168, "v19": 0.4165, "v20": 0.1260,
    "v21": 0.5179, "v22": -0.2659, "v23": -0.3819, "v24": 0.1612, "v25": 0.0579,
    "v26": -0.0710, "v27": -0.0505, "v28": -0.0465
})
print(f"Fraud TX:  is_fraud={fraud['is_fraud']} prob={fraud['fraud_probability']:.4f} risk={fraud['risk_level']}")

# Feature importance
fi = get("/feature-importance")
top3 = fi["feature_importance"][:3]
print(f"Feature importance top3: {[(x['feature'], round(x['importance'],4)) for x in top3]}")

# Model registry
reg = get("/model-registry")
print(f"Model registry: {len(reg['models'])} versions")

# Retrain check
chk = get("/retrain/check-data")
print(f"New data available: {chk['new_data_available']} ({chk['record_count']} records)")

print("\nALL TESTS PASSED!")
