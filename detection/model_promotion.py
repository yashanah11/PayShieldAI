import json
import shutil
from pathlib import Path

import joblib
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from generation.fraud_injector import generate_fraud_dataset
from generation.generator import FEATURES


MODEL_PATH = Path("models/xgboost_detector.joblib")
BACKUP_PATH = Path("models/xgboost_detector.backup.joblib")


def train_candidate(seed=42):
    df = generate_fraud_dataset(
        10000,
        fraud_rate=0.05,
        seed=seed,
    )

    model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=seed,
    )

    model.fit(df[FEATURES], df["is_fraud"])
    return model


def evaluate(model, seed=10042):
    if not hasattr(model, "predict_proba"):
        raise TypeError("Candidate model must provide predict_proba()")

    df = generate_fraud_dataset(
        5000,
        fraud_rate=0.05,
        seed=seed,
    )

    probabilities = model.predict_proba(df[FEATURES])[:, 1]

    return roc_auc_score(
        df["is_fraud"],
        probabilities,
    )


def promote_if_better(candidate, old_auc, candidate_auc):
    if not hasattr(candidate, "predict_proba"):
        return False

    if candidate_auc < old_auc:
        return False

    if not MODEL_PATH.exists():
        return False

    shutil.copy2(
        MODEL_PATH,
        BACKUP_PATH,
    )

    joblib.dump(
        candidate,
        MODEL_PATH,
    )

    return True


def run_model_promotion(seed=42):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model: {MODEL_PATH}"
        )

    current_model = joblib.load(MODEL_PATH)

    old_auc = evaluate(
        current_model,
        seed=seed + 1000,
    )

    candidate = train_candidate(seed)

    candidate_auc = evaluate(
        candidate,
        seed=seed + 1000,
    )

    promoted = promote_if_better(
        candidate,
        old_auc,
        candidate_auc,
    )

    result = {
        "old_roc_auc": float(old_auc),
        "candidate_roc_auc": float(candidate_auc),
        "promoted": bool(promoted),
        "rollback_available": bool(
            BACKUP_PATH.exists()
        ),
    }

    Path("evaluation").mkdir(exist_ok=True)

    with open(
        "evaluation/model_promotion.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    print("MODEL PROMOTION: OK")
    print(f"OLD ROC-AUC: {old_auc:.4f}")
    print(f"CANDIDATE ROC-AUC: {candidate_auc:.4f}")
    print("PROMOTED:", promoted)
    print("ROLLBACK AVAILABLE:", BACKUP_PATH.exists())


if __name__ == "__main__":
    run_model_promotion()
