"""Script to run initial model training."""
import sys
sys.path.insert(0, '.')

from app.ml.pipeline import run_training_pipeline
from app.core.config import settings
from app.db.database import init_db, SessionLocal

print("Initializing database...")
init_db()

db = SessionLocal()
print("Starting training pipeline...")
result = run_training_pipeline(settings.TRAINING_DATA_FILE, db=db)
db.close()

print("\n=== TRAINING COMPLETE ===")
print(f"Version:     {result['version']}")
print(f"Model:       {result['model_type']}")
print(f"ROC-AUC:     {result['metrics']['roc_auc']:.4f}")
print(f"F1 Score:    {result['metrics']['f1_score']:.4f}")
print(f"Recall:      {result['metrics']['recall']:.4f}")
print(f"Precision:   {result['metrics']['precision']:.4f}")
print(f"Accuracy:    {result['metrics']['accuracy']:.4f}")
print(f"Dataset:     {result['dataset_size']} records")
print(f"\nAll Model Comparison:")
for name, m in result['all_model_metrics'].items():
    print(f"  {name:25s}  ROC-AUC={m['roc_auc']:.4f}  F1={m['f1_score']:.4f}")

print(f"\nTop 5 Important Features:")
sorted_fi = sorted(result['feature_importance'].items(), key=lambda x: x[1], reverse=True)[:5]
for feat, imp in sorted_fi:
    print(f"  {feat}: {imp:.4f}")
