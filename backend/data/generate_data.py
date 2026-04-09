"""
Synthetic Credit Card Fraud Dataset Generator
Generates realistic transaction data mimicking the Kaggle Credit Card Fraud dataset.
Features: Time, V1-V28 (PCA-transformed), Amount, Class (0=normal, 1=fraud)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os


def generate_fraud_dataset(n_normal: int = 14500, n_fraud: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic credit card transaction dataset with class imbalance."""
    np.random.seed(seed)

    # --- Normal transactions ---
    normal_v_means = np.zeros(28)
    normal_v_cov = np.eye(28) * 0.5 + np.random.uniform(-0.1, 0.1, (28, 28))
    np.fill_diagonal(normal_v_cov, 1.0)
    normal_v_cov = (normal_v_cov + normal_v_cov.T) / 2  # symmetrize
    np.fill_diagonal(normal_v_cov, 1.0)

    V_normal = np.random.multivariate_normal(normal_v_means, normal_v_cov, n_normal)
    amount_normal = np.random.lognormal(mean=3.5, sigma=1.5, size=n_normal)
    amount_normal = np.clip(amount_normal, 0.5, 25000)
    time_normal = np.sort(np.random.uniform(0, 172800, n_normal))

    # --- Fraud transactions ---
    fraud_v_means = np.array([
        -3.2, 2.8, -2.1, 1.9, -1.5, 0.8, -2.4, 0.5,
        -1.8, -3.5, 2.2, -1.7, 2.9, -4.1, 1.1, -0.9,
        -0.5, 1.3, -0.7, 0.3, 0.8, -0.4, 0.2, -0.6,
        0.4, -0.3, 0.7, -0.5
    ])
    fraud_v_cov = np.eye(28) * 1.2
    V_fraud = np.random.multivariate_normal(fraud_v_means, fraud_v_cov, n_fraud)
    # Add noise to some fraud V features for realism
    V_fraud[:, 0] += np.random.normal(-1, 0.5, n_fraud)
    V_fraud[:, 3] += np.random.normal(1, 0.4, n_fraud)
    V_fraud[:, 9] += np.random.normal(-1.5, 0.6, n_fraud)

    # Fraud amounts: mix of small and large
    amount_fraud_small = np.random.uniform(0.5, 50, n_fraud // 2)
    amount_fraud_large = np.random.lognormal(mean=5.5, sigma=1.0, size=n_fraud - n_fraud // 2)
    amount_fraud = np.concatenate([amount_fraud_small, amount_fraud_large])
    np.random.shuffle(amount_fraud)
    amount_fraud = np.clip(amount_fraud, 0.5, 25000)
    time_fraud = np.random.uniform(0, 172800, n_fraud)

    # --- Assemble DataFrames ---
    cols = [f"V{i}" for i in range(1, 29)]

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

    # Reorder columns to match original dataset
    feature_cols = ["Time"] + cols + ["Amount", "Class"]
    df = df[feature_cols]

    return df


def generate_new_transactions(n_normal: int = 2700, n_fraud: int = 300, seed: int = 99) -> pd.DataFrame:
    """Generate a smaller batch of new transactions to simulate data drift & retraining trigger."""
    np.random.seed(seed)

    # Slightly different distributions to simulate data drift
    normal_v_means = np.random.normal(0, 0.1, 28)  # slight shift
    normal_v_cov = np.eye(28) * 1.1
    V_normal = np.random.multivariate_normal(normal_v_means, normal_v_cov, n_normal)
    amount_normal = np.random.lognormal(mean=3.7, sigma=1.4, size=n_normal)
    amount_normal = np.clip(amount_normal, 0.5, 30000)
    time_normal = np.sort(np.random.uniform(172800, 345600, n_normal))  # next 2 days

    # Fraud with slightly stronger patterns (new fraud technique)
    fraud_v_means = np.array([
        -3.8, 3.2, -2.5, 2.3, -1.9, 1.1, -2.8, 0.7,
        -2.2, -4.0, 2.6, -2.1, 3.3, -4.6, 1.4, -1.2,
        -0.8, 1.6, -1.0, 0.5, 1.0, -0.6, 0.3, -0.8,
        0.6, -0.5, 0.9, -0.7
    ])
    fraud_v_cov = np.eye(28) * 1.0
    V_fraud = np.random.multivariate_normal(fraud_v_means, fraud_v_cov, n_fraud)
    amount_fraud = np.concatenate([
        np.random.uniform(0.5, 30, n_fraud // 2),
        np.random.lognormal(mean=6.0, sigma=0.8, size=n_fraud - n_fraud // 2)
    ])
    np.random.shuffle(amount_fraud)
    amount_fraud = np.clip(amount_fraud, 0.5, 30000)
    time_fraud = np.random.uniform(172800, 345600, n_fraud)

    cols = [f"V{i}" for i in range(1, 29)]

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
    df = df[feature_cols]

    return df


def main():
    data_dir = Path(__file__).parent
    data_dir.mkdir(exist_ok=True)

    print("Generating primary training dataset...")
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
