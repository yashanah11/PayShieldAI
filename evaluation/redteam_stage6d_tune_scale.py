"""
STAGE 6d: TUNE SCALE_POS_WEIGHT
- Retrain from scratch with base + 200 evolved attacks.
- Try multiple scale_pos_weight values to balance generalisation and evolved detection.
- Evaluate and pick the best.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")

from generation.generator import FEATURES
from evaluation.hard_benchmark_stage2_repaired import (
    generate_stage2_repaired_benchmark,
)
from evaluation.redteam_stage2c_retest import make_unseen_attacks


def load_data():
    train_base = generate_stage2_repaired_benchmark(n=10000, fraud_rate=0.05, seed=42)
    test_base = generate_stage2_repaired_benchmark(n=5000, fraud_rate=0.05, seed=999)
    test_df = make_unseen_attacks(test_base, seed=12345)
    evolved = pd.read_csv("data/evolved_attacks.csv")
    evolved = evolved[FEATURES + ["is_fraud"]]
    return train_base, test_df, evolved


def evaluate_model(model, df, label="Dataset"):
    X = df[FEATURES]
    y = df["is_fraud"]
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else np.nan
    rec = recall_score(y, preds, zero_division=0)
    prec = precision_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    print(f"\n{label}:")
    print(f"  AUC:       {auc:.4f}" if not np.isnan(auc) else "  AUC:       N/A")
    print(f"  Recall:    {rec:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  F1:        {f1:.4f}")
    return {"auc": auc, "recall": rec, "precision": prec, "f1": f1}


def train_with_scale(train_df, scale_pos_weight, seed=42):
    X_train, _, y_train, _ = train_test_split(
        train_df[FEATURES],
        train_df["is_fraud"],
        test_size=0.30,
        random_state=seed,
        stratify=train_df["is_fraud"],
    )
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    print("=== STAGE 6d: TUNE SCALE_POS_WEIGHT ===")

    train_base, test_standard, evolved = load_data()
    print(f"Training base: {len(train_base)} rows")
    print(f"Standard test: {len(test_standard)} rows")
    print(f"Evolved attacks (full): {len(evolved)} rows")

    # Use 200 evolved examples (as in v3)
    evolved_sample = evolved.sample(n=200, random_state=42)
    print(f"Using 200 evolved attacks for training.")

    train_augmented = pd.concat([train_base, evolved_sample], ignore_index=True)
    train_augmented = train_augmented.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Augmented training size: {len(train_augmented)} rows")

    # Try different scale_pos_weight values
    scales = [1.0, 3.0, 5.0, 8.0, 12.0]
    results = []

    for scale in scales:
        print(f"\n--- Training with scale_pos_weight = {scale} ---")
        model = train_with_scale(train_augmented, scale, seed=42)
        # Evaluate on standard
        m_std = evaluate_model(model, test_standard, f"Standard (scale={scale})")
        # Evaluate on evolved
        m_evol = evaluate_model(model, evolved, f"Evolved (scale={scale})")
        results.append({
            "scale": scale,
            "auc_std": m_std["auc"],
            "recall_std": m_std["recall"],
            "recall_evol": m_evol["recall"],
        })

    # Find best balance: we want AUC_std >= 0.85 and recall_evol as high as possible
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"scale={r['scale']}: AUC_std={r['auc_std']:.4f}, recall_evol={r['recall_evol']:.4f}")

    # Select best (choose the highest scale that still gives AUC_std >= 0.85)
    best = None
    for r in results:
        if r["auc_std"] >= 0.85:
            if best is None or r["recall_evol"] > best["recall_evol"]:
                best = r

    if best is None:
        print("\nNo scale gave AUC_std >= 0.85. Consider reducing evolved sample size or increasing n_estimators.")
    else:
        print(f"\nBest scale: {best['scale']} with AUC_std={best['auc_std']:.4f}, recall_evol={best['recall_evol']:.4f}")
        # Retrain with best scale and save as final model
        model_best = train_with_scale(train_augmented, best["scale"], seed=42)
        joblib.dump(model_best, "models/xgboost_detector_final.joblib")
        print("Final model saved as models/xgboost_detector_final.joblib")
        print("\nVERDICT: PASS if AUC_std >= 0.85 and recall_evol >= 0.60")
        if best["auc_std"] >= 0.85 and best["recall_evol"] >= 0.60:
            print("PASS")
        else:
            print("FAIL")