"""
Stage 10: Adversarial Retraining
PayShieldAI System

Trains a NEW v6_robust XGBoost model using the Stage 10 adversarial
training dataset.

IMPORTANT:
- Does NOT modify v1
- Does NOT modify v3
- Does NOT modify ensemble_config.joblib
- Does NOT overwrite legacy v4/v5 models
- Uses ONLY the canonical 7-feature schema
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    confusion_matrix,
)

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "stage10_adversarial_train.csv",
)

MODEL_OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "xgboost_detector_v6_robust.joblib",
)


# ============================================================
# LOCKED FEATURE SCHEMA
# ============================================================

FEATURES = [
    "amount",
    "hour",
    "velocity_1h",
    "velocity_24h",
    "device_age_days",
    "distance_km",
    "merchant_risk",
]

TARGET = "is_fraud"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

# New model configuration.
# Slightly deeper trees allow learning feature interactions.
N_ESTIMATORS = 400
MAX_DEPTH = 6
LEARNING_RATE = 0.05
SUBSAMPLE = 0.85
COLSAMPLE_BYTREE = 0.90

# ============================================================
# MAIN
# ============================================================


def train_v6_robust():

    print("=" * 70)
    print("PAYSHIELD-AI: STAGE 10 ADVERSARIAL RETRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Verify dataset
    # --------------------------------------------------------

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Stage 10 dataset not found:\n{DATA_PATH}"
        )

    print("\n[INFO] Loading Stage 10 adversarial dataset...")
    df = pd.read_csv(DATA_PATH)

    print(f"[INFO] Dataset rows    : {len(df)}")
    print(f"[INFO] Dataset columns : {len(df.columns)}")

    # --------------------------------------------------------
    # 2. Strict schema validation
    # --------------------------------------------------------

    expected_columns = FEATURES + [TARGET]

    missing = [
        col for col in expected_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"CRITICAL: Missing required columns: {missing}"
        )

    extra = [
        col for col in df.columns
        if col not in expected_columns
    ]

    if extra:
        print(
            f"[WARNING] Extra columns detected: {extra}"
        )

    # Use ONLY canonical features.
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    # --------------------------------------------------------
    # 3. Dataset integrity
    # --------------------------------------------------------

    if y.nunique() != 2:
        raise ValueError(
            "CRITICAL: Training dataset must contain both "
            "benign and fraud classes."
        )

    fraud_count = int((y == 1).sum())
    benign_count = int((y == 0).sum())

    print("\n[INFO] Class distribution:")
    print(f"  Benign : {benign_count}")
    print(f"  Fraud  : {fraud_count}")
    print(
        f"  Fraud rate: {fraud_count / len(y) * 100:.2f}%"
    )

    # Check for NaN / infinite values.
    if X.isnull().any().any():
        raise ValueError(
            "CRITICAL: NaN values found in training features."
        )

    if np.isinf(X.to_numpy()).any():
        raise ValueError(
            "CRITICAL: Infinite values found in training features."
        )

    # --------------------------------------------------------
    # 4. Train/test split
    # --------------------------------------------------------

    print("\n[INFO] Creating stratified train/test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"[INFO] Training samples : {len(X_train)}")
    print(f"[INFO] Testing samples  : {len(X_test)}")

    # --------------------------------------------------------
    # 5. Calculate class weighting
    # --------------------------------------------------------

    # scale_pos_weight compensates for the 90/10 distribution.
    train_benign = int((y_train == 0).sum())
    train_fraud = int((y_train == 1).sum())

    scale_pos_weight = (
        train_benign / train_fraud
    )

    print(
        f"\n[INFO] scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    # --------------------------------------------------------
    # 6. Build NEW v6_robust model
    # --------------------------------------------------------

    print("\n[INFO] Initializing v6_robust XGBoost...")

    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,

        objective="binary:logistic",
        eval_metric="auc",

        scale_pos_weight=scale_pos_weight,

        random_state=RANDOM_STATE,
        n_jobs=-1,

        tree_method="hist",
    )

    # --------------------------------------------------------
    # 7. Train
    # --------------------------------------------------------

    print("\n[INFO] Training v6_robust...")
    print("[INFO] Locked v1/v3 models will NOT be modified.")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    print("[SUCCESS] Training completed.")

    # --------------------------------------------------------
    # 8. Evaluate internal holdout
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("V6_ROBUST INTERNAL HOLDOUT EVALUATION")
    print("-" * 70)

    probabilities = model.predict_proba(X_test)[:, 1]

    # Use the current Stage 6C threshold ONLY for reporting.
    # This does not modify ensemble_config.joblib.
    threshold = 0.30

    predictions = (
        probabilities >= threshold
    ).astype(int)

    auc = roc_auc_score(
        y_test,
        probabilities,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    print(f"ROC-AUC  : {auc:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"Precision: {precision:.4f}")

    print("\nConfusion Matrix:")
    print(f"  True Negatives : {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives : {tp}")

    # --------------------------------------------------------
    # 9. Feature importance
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("V6_ROBUST FEATURE IMPORTANCE")
    print("-" * 70)

    importances = model.feature_importances_

    importance_pairs = sorted(
        zip(FEATURES, importances),
        key=lambda x: x[1],
        reverse=True,
    )

    for feature, importance in importance_pairs:
        print(
            f"  {feature:<18}: "
            f"{importance:.4f}"
        )

    # --------------------------------------------------------
    # 10. Verify model schema
    # --------------------------------------------------------

    if hasattr(model, "feature_names_in_"):

        model_features = list(
            model.feature_names_in_
        )

        if model_features != FEATURES:
            raise ValueError(
                "CRITICAL: v6_robust feature schema mismatch.\n"
                f"Expected: {FEATURES}\n"
                f"Actual  : {model_features}"
            )

        print(
            "\n[INFO] v6_robust feature schema verified."
        )

    # --------------------------------------------------------
    # 11. Save safely
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(MODEL_OUTPUT_PATH),
        exist_ok=True,
    )

    # Safety check: never overwrite locked models.
    locked_paths = [
        os.path.join(
            PROJECT_ROOT,
            "models",
            "xgboost_detector_retrained.joblib",
        ),
        os.path.join(
            PROJECT_ROOT,
            "models",
            "xgboost_detector_retrained_v3.joblib",
        ),
        os.path.join(
            PROJECT_ROOT,
            "models",
            "ensemble_config.joblib",
        ),
    ]

    if os.path.abspath(MODEL_OUTPUT_PATH) in [
        os.path.abspath(p)
        for p in locked_paths
    ]:
        raise RuntimeError(
            "CRITICAL: Refusing to overwrite locked model/config."
        )

    print("\n[INFO] Saving v6_robust model...")

    joblib.dump(
        model,
        MODEL_OUTPUT_PATH,
    )

    # --------------------------------------------------------
    # 12. Verify saved model
    # --------------------------------------------------------

    if not os.path.exists(MODEL_OUTPUT_PATH):
        raise RuntimeError(
            "CRITICAL: Model save failed."
        )

    saved_model = joblib.load(
        MODEL_OUTPUT_PATH
    )

    # Verify prediction works after reload.
    saved_probabilities = (
        saved_model.predict_proba(X_test)[:, 1]
    )

    saved_auc = roc_auc_score(
        y_test,
        saved_probabilities,
    )

    print(
        f"[INFO] Reload verification AUC: "
        f"{saved_auc:.4f}"
    )

    print("\n" + "=" * 70)
    print("STAGE 10 RETRAINING COMPLETE")
    print("=" * 70)

    print(
        f"[SUCCESS] v6_robust saved to:\n"
        f"          {MODEL_OUTPUT_PATH}"
    )

    print("\nModel summary:")
    print(f"  Samples       : {len(df)}")
    print(f"  Fraud samples : {fraud_count}")
    print(f"  Estimators    : {N_ESTIMATORS}")
    print(f"  Max depth     : {MAX_DEPTH}")
    print(f"  Learning rate : {LEARNING_RATE}")

    print("\nInternal holdout:")
    print(f"  AUC           : {auc:.4f}")
    print(f"  Recall        : {recall:.4f}")
    print(f"  Precision     : {precision:.4f}")

    print("\n[IMPORTANT]")
    print("  v1                  : UNCHANGED")
    print("  v3                  : UNCHANGED")
    print("  ensemble_config     : UNCHANGED")
    print("  legacy v4/v5        : UNCHANGED")

    print("=" * 70)


if __name__ == "__main__":
    train_v6_robust()