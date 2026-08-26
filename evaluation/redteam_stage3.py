import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from generation.generator import FEATURES
from evaluation.hard_benchmark import generate_hard_benchmark


def augment_with_attack_families(df, seed=42):
    """Augment training data with pure attack-family anomalies."""
    rng = np.random.default_rng(seed)
    base_size = len(df)
    
    # Separate benign rows (we only augment from benign, to create pure anomalies)
    benign = df[df["is_fraud"] == 0].copy()
    
    # Number of examples to add for each family (5% of total)
    n_add = int(0.05 * base_size)
    
    augmented_dfs = [df]  # start with original
    
    # Family 1: Velocity
    idx_vel = rng.choice(benign.index, size=n_add, replace=False)
    vel_examples = benign.loc[idx_vel].copy()
    vel_examples["velocity_1h"] += rng.integers(2, 5, len(vel_examples))
    vel_examples["velocity_24h"] += rng.integers(3, 10, len(vel_examples))
    vel_examples["is_fraud"] = 1
    augmented_dfs.append(vel_examples)
    
    # Family 2: Geographic
    # Use a different subset to avoid overlap with velocity (optional, but clean)
    remaining = benign.drop(idx_vel)  # remove already used indices
    idx_geo = rng.choice(remaining.index, size=n_add, replace=False)
    geo_examples = remaining.loc[idx_geo].copy()
    geo_examples["distance_km"] += rng.uniform(30, 120, len(geo_examples))
    geo_examples["is_fraud"] = 1
    augmented_dfs.append(geo_examples)
    
    # Family 3: Coordinated
    remaining2 = remaining.drop(idx_geo)
    idx_coord = rng.choice(remaining2.index, size=n_add, replace=False)
    coord_examples = remaining2.loc[idx_coord].copy()
    coord_examples["amount"] *= rng.uniform(1.5, 3.0, len(coord_examples))
    coord_examples["velocity_24h"] += rng.integers(2, 8, len(coord_examples))
    coord_examples["distance_km"] += rng.uniform(20, 100, len(coord_examples))
    coord_examples["is_fraud"] = 1
    augmented_dfs.append(coord_examples)
    
    # Combine all and shuffle
    df_augmented = pd.concat(augmented_dfs, ignore_index=True)
    df_augmented = df_augmented.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return df_augmented


def train_candidate(seed=42):
    # Generate base training data (statistical fraud only)
    df = generate_hard_benchmark(
        n=10000,
        fraud_rate=0.05,
        seed=seed,
    )
    
    # --- REPAIR: Augment with all three attack families ---
    df_augmented = augment_with_attack_families(df, seed=seed)
    
    # Train-test split on augmented data
    X_train, _, y_train, _ = train_test_split(
        df_augmented[FEATURES],
        df_augmented["is_fraud"],
        test_size=0.30,
        random_state=seed,
        stratify=df_augmented["is_fraud"],
    )
    
    # Same hyperparameters (we only change data)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
    )
    
    model.fit(X_train, y_train)
    return model


def make_attack_family(base, family, seed=12345):
    rng = np.random.default_rng(seed)
    
    df = base.copy()
    
    fraud_indices = rng.choice(
        df.index,
        size=max(1, int(len(df) * 0.05)),
        replace=False,
    )
    
    df["is_fraud"] = 0
    df.loc[fraud_indices, "is_fraud"] = 1
    
    if family == "velocity":
        idx = fraud_indices
        df.loc[idx, "velocity_1h"] += rng.integers(2, 5, len(idx))
        df.loc[idx, "velocity_24h"] += rng.integers(3, 10, len(idx))
    
    elif family == "geographic":
        idx = fraud_indices
        df.loc[idx, "distance_km"] += rng.uniform(30, 120, len(idx))
    
    elif family == "coordinated":
        idx = fraud_indices
        df.loc[idx, "amount"] *= rng.uniform(1.5, 3.0, len(idx))
        df.loc[idx, "velocity_24h"] += rng.integers(2, 8, len(idx))
        df.loc[idx, "distance_km"] += rng.uniform(20, 100, len(idx))
    
    return df


def evaluate(model, df):
    X = df[FEATURES]
    y = df["is_fraud"]
    
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    
    return {
        "auc": roc_auc_score(y, probabilities),
        "recall": recall_score(y, predictions, zero_division=0),
    }


if __name__ == "__main__":
    print("=== RED-TEAM STAGE 3: ATTACK FAMILY DIAGNOSIS ===")
    
    model = train_candidate()
    
    base = generate_hard_benchmark(
        n=5000,
        fraud_rate=0.05,
        seed=999,
    )
    
    families = ["velocity", "geographic", "coordinated"]
    results = {}
    
    for family in families:
        df = make_attack_family(base, family)
        metrics = evaluate(model, df)
        results[family] = metrics
        
        print()
        print(f"ATTACK FAMILY: {family.upper()}")
        print(f"ROC-AUC : {metrics['auc']:.4f}")
        print(f"Recall  : {metrics['recall']:.4f}")
    
    print()
    print("=== DIAGNOSIS ===")
    weakest = min(results, key=lambda x: results[x]["auc"])
    print(f"Weakest family: {weakest}")
    
    if results[weakest]["auc"] < 0.80:
        print("STATUS: WEAKNESS CONFIRMED")
    else:
        print("STATUS: ALL FAMILIES PASS")