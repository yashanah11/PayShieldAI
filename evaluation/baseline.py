import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from generation.fraud_injector import generate_fraud_dataset
from generation.generator import FEATURES


def evaluate_detector(n=10000, seed=123):
    df = generate_fraud_dataset(n, fraud_rate=0.05, seed=seed)

    model = joblib.load("models/xgboost_detector.joblib")

    X = df[FEATURES]
    y = df["is_fraud"]

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()

    metrics = {
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    Path("evaluation").mkdir(exist_ok=True)

    with open("evaluation/baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("EVALUATION: OK")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")


if __name__ == "__main__":
    evaluate_detector()
