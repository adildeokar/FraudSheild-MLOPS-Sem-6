"""
Synthetic Credit Card Fraud Dataset Generator
Generates transaction-like data (Time, V1–V28, Amount, Class) with deliberate
class overlap so supervised models show realistic sub-perfect metrics — not toy 100% scores.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os


def _sym_psd(cov: np.ndarray, floor: float = 0.15) -> np.ndarray:
    cov = (cov + cov.T) / 2.0
    w, v = np.linalg.eigh(cov)
    w = np.clip(w, floor, None)
    return (v @ np.diag(w) @ v.T).astype(np.float64)


def generate_fraud_dataset(n_normal: int = 14500, n_fraud: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic data with heavy overlap between classes (realistic fraud is subtle).
    """
    rng = np.random.default_rng(seed)
    cols = [f"V{i}" for i in range(1, 29)]

    # Shared background covariance (fraud hides in the bulk of normal PCA space)
    base_cov = np.eye(28, dtype=np.float64) * 0.85
    for i in range(28):
        for j in range(i + 1, 28):
            c = rng.normal(0, 0.12)
            base_cov[i, j] = base_cov[j, i] = c
    base_cov = _sym_psd(base_cov)

    # Normal: centered near 0 with shared structure
    normal_mean = rng.normal(0, 0.08, 28)
    V_normal = rng.multivariate_normal(normal_mean, base_cov, n_normal)

    # Fraud: mixture of (a) subtle shift + (b) samples drawn from normal-like cloud (hard fraud)
    fraud_shift = np.array([
        -0.9, 0.55, -0.45, 0.5, -0.35, 0.25, -0.55, 0.15,
        -0.4, -0.65, 0.45, -0.35, 0.5, -0.75, 0.25, -0.2,
        -0.15, 0.22, -0.18, 0.1, 0.18, -0.12, 0.08, -0.14,
        0.1, -0.08, 0.12, -0.1,
    ], dtype=np.float64)
    fraud_cov = _sym_psd(base_cov + np.eye(28) * 0.35)

    n_subtle = int(n_fraud * 0.55)
    n_mixed = n_fraud - n_subtle
    V_subtle = rng.multivariate_normal(fraud_shift, fraud_cov, n_subtle)
    # Mixed: mostly normal distribution with slight fraud nudge (overlap / false separability)
    V_mixed = rng.multivariate_normal(normal_mean * 0.6 + fraud_shift * 0.35, base_cov * 1.15, n_mixed)
    V_fraud = np.vstack([V_subtle, V_mixed])
    rng.shuffle(V_fraud)

    # Global jitter on all V (reduces linear separability)
    jitter = rng.normal(0, 0.22, V_normal.shape)
    V_normal = V_normal + jitter
    V_fraud = V_fraud + rng.normal(0, 0.22, V_fraud.shape)

    amount_normal = rng.lognormal(mean=3.4, sigma=1.45, size=n_normal)
    amount_normal = np.clip(amount_normal, 0.5, 25000)
    time_normal = np.sort(rng.uniform(0, 172800, n_normal))

    # Fraud amounts overlap normal (many frauds are small amounts)
    amt_mix = rng.random(n_fraud)
    amount_fraud = np.where(
        amt_mix < 0.55,
        rng.lognormal(mean=3.2, sigma=1.35, size=n_fraud),
        rng.lognormal(mean=4.8, sigma=1.05, size=n_fraud),
    )
    amount_fraud = np.clip(amount_fraud, 0.5, 25000)
    time_fraud = rng.uniform(0, 172800, n_fraud)

    df_normal = pd.DataFrame(V_normal, columns=cols)
    df_normal["Time"] = time_normal
    df_normal["Amount"] = amount_normal
    df_normal["Class"] = 0

    df_fraud = pd.DataFrame(V_fraud, columns=cols)
    df_fraud["Time"] = time_fraud
    df_fraud["Amount"] = amount_fraud
    df_fraud["Class"] = 1

    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    feature_cols = ["Time"] + cols + ["Amount", "Class"]
    return df[feature_cols]


def generate_new_transactions(n_normal: int = 2700, n_fraud: int = 300, seed: int = 99) -> pd.DataFrame:
    """Smaller batch with drift + overlap (retraining demo)."""
    rng = np.random.default_rng(seed)
    cols = [f"V{i}" for i in range(1, 29)]

    base_cov = np.eye(28) * 0.95
    for i in range(28):
        for j in range(i + 1, 28):
            c = rng.normal(0, 0.1)
            base_cov[i, j] = base_cov[j, i] = c
    base_cov = _sym_psd(base_cov)

    normal_mean = rng.normal(0, 0.12, 28)
    V_normal = rng.multivariate_normal(normal_mean, base_cov, n_normal)

    fraud_shift = rng.normal(0, 0.35, 28) + np.array(
        [-0.7, 0.5, -0.35, 0.45, -0.28, 0.2, -0.45, 0.12] + [0.0] * 20, dtype=np.float64
    )[:28]
    fraud_cov = _sym_psd(base_cov + np.eye(28) * 0.4)
    n_subtle = int(n_fraud * 0.5)
    V_subtle = rng.multivariate_normal(fraud_shift, fraud_cov, n_subtle)
    V_mixed = rng.multivariate_normal(normal_mean * 0.55 + fraud_shift * 0.4, base_cov * 1.12, n_fraud - n_subtle)
    V_fraud = np.vstack([V_subtle, V_mixed])
    rng.shuffle(V_fraud)

    V_normal += rng.normal(0, 0.24, V_normal.shape)
    V_fraud += rng.normal(0, 0.24, V_fraud.shape)

    amount_normal = rng.lognormal(mean=3.55, sigma=1.42, size=n_normal)
    amount_normal = np.clip(amount_normal, 0.5, 30000)
    time_normal = np.sort(rng.uniform(172800, 345600, n_normal))

    amount_fraud = np.where(
        rng.random(n_fraud) < 0.5,
        rng.lognormal(mean=3.25, sigma=1.3, size=n_fraud),
        rng.lognormal(mean=5.0, sigma=1.0, size=n_fraud),
    )
    amount_fraud = np.clip(amount_fraud, 0.5, 30000)
    time_fraud = rng.uniform(172800, 345600, n_fraud)

    df_normal = pd.DataFrame(V_normal, columns=cols)
    df_normal["Time"] = time_normal
    df_normal["Amount"] = amount_normal
    df_normal["Class"] = 0

    df_fraud = pd.DataFrame(V_fraud, columns=cols)
    df_fraud["Time"] = time_fraud
    df_fraud["Amount"] = amount_fraud
    df_fraud["Class"] = 1

    df = pd.concat([df_normal, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    feature_cols = ["Time"] + cols + ["Amount", "Class"]
    return df[feature_cols]


def main():
    data_dir = Path(__file__).parent
    data_dir.mkdir(exist_ok=True)

    print("Generating primary training dataset (realistic overlap)...")
    df_train = generate_fraud_dataset(n_normal=14500, n_fraud=500)
    train_path = data_dir / "creditcard.csv"
    df_train.to_csv(train_path, index=False)
    print(f"  Saved {len(df_train)} transactions to {train_path}")
    print(f"  Normal: {(df_train['Class']==0).sum()} | Fraud: {(df_train['Class']==1).sum()}")
    print(f"  Fraud rate: {df_train['Class'].mean()*100:.2f}%")

    print("\nGenerating new transaction batch (for retraining demo)...")
    df_new = generate_new_transactions(n_normal=2700, n_fraud=300)
    new_path = data_dir / "New_transactions.csv"
    df_new.to_csv(new_path, index=False)
    print(f"  Saved {len(df_new)} transactions to {new_path}")
    print(f"  Normal: {(df_new['Class']==0).sum()} | Fraud: {(df_new['Class']==1).sum()}")

    print("\nData generation complete!")
    return str(train_path), str(new_path)


if __name__ == "__main__":
    main()
