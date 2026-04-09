"""Test the auto-retraining pipeline."""
import urllib.request, json, sys

BASE = "http://localhost:8000/api"

def post(path):
    req = urllib.request.Request(f"{BASE}{path}", data=b"{}",
        headers={"Content-Type": "application/json"}, method="POST")
    res = urllib.request.urlopen(req, timeout=120)
    return json.loads(res.read())

print("Triggering auto-retraining pipeline...")
print("(This trains LR + RF + XGBoost on 18,000 combined records)")
print()

result = post("/retrain")

print("=== RETRAINING RESULT ===")
print(f"Status:   {result['status']}")
print(f"Improved: {result['improved']}")
print(f"Improvement: {result['improvement']:+.4f}")
print()
print(f"Old Model: {result['old_model']['version']} ({result['old_model']['model_type']}) - ROC-AUC={result['old_model']['roc_auc']:.4f}")
print(f"New Model: {result['new_model']['version']} ({result['new_model']['model_type']}) - ROC-AUC={result['new_model']['roc_auc']:.4f}")
print()
print("All models in competition:")
for name, m in result['all_model_metrics'].items():
    print(f"  {name:25s}  ROC-AUC={m['roc_auc']:.4f}  F1={m['f1_score']:.4f}")
print()
print(f"Dataset size: {result['dataset_size']:,} records ({result['fraud_count']:,} fraud)")
print()
print(f"Decision: {result['message']}")
