import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from evaluation.hard_benchmark import generate_hard_benchmark
from generation.generator import FEATURES


OUTPUT = Path("evaluation/hard_benchmark_experiment.json")


def metrics(model, X, y):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y, predictions, zero_division=0)
        ),
    }


def train_candidate(X, y, seed=42):
    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
    )

    model.fit(X, y)

    return model


def run_experiment(n=10000, fraud_rate=0.05, seed=42):
    df = generate_hard_benchmark(
        n=n,
        fraud_rate=fraud_rate,
        seed=seed,
    )

    X = df[FEATURES]
    y = df["is_fraud"]

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )

    # Existing production model.
    baseline_model = joblib.load(
        "models/xgboost_detector.joblib"
    )

    baseline_metrics = metrics(
        baseline_model,
        X_holdout,
        y_holdout,
    )

    # Mine hard examples ONLY from the training partition.
    train_probabilities = baseline_model.predict_proba(
        X_train
    )[:, 1]

    hard_mask = (
        ((y_train.to_numpy() == 1) &
         (train_probabilities < 0.5))
        |
        ((y_train.to_numpy() == 0) &
         (train_probabilities >= 0.5))
    )

    hard_examples = X_train.loc[hard_mask]
    hard_labels = y_train.loc[hard_mask]

    # Candidate training data = original training data
    # plus additional weight on hard examples.
    X_candidate = X_train.copy()
    y_candidate = y_train.copy()

    if len(hard_examples) > 0:
        X_candidate = np.concatenate(
            [
                X_candidate.to_numpy(),
                hard_examples.to_numpy(),
            ]
        )

        y_candidate = np.concatenate(
            [
                y_candidate.to_numpy(),
                hard_labels.to_numpy(),
            ]
        )

    candidate_model = train_candidate(
        X_candidate,
        y_candidate,
        seed=seed,
    )

    candidate_metrics = metrics(
        candidate_model,
        X_holdout,
        y_holdout,
    )

    delta = (
        candidate_metrics["roc_auc"]
        - baseline_metrics["roc_auc"]
    )

    result = {
        "experiment": "hard_benchmark_self_improvement",
        "data_type": "synthetic",
        "seed": seed,
        "transactions": n,
        "fraud_rate": fraud_rate,
        "train_size": len(X_train),
        "holdout_size": len(X_holdout),
        "hard_examples_found": int(len(hard_examples)),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "roc_auc_delta": float(delta),
        "candidate_better": bool(delta > 0),
        "promotion_recommended": bool(delta > 0),
        "holdout_used_for_training": False,
    }

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=== HARD BENCHMARK EXPERIMENT ===")
    print(f"Training transactions : {len(X_train)}")
    print(f"Holdout transactions  : {len(X_holdout)}")
    print(f"Hard examples found   : {len(hard_examples)}")
    print()
    print("BASELINE")
    print(f"ROC-AUC   : {baseline_metrics['roc_auc']:.4f}")
    print(f"Precision : {baseline_metrics['precision']:.4f}")
    print(f"Recall    : {baseline_metrics['recall']:.4f}")
    print(f"F1        : {baseline_metrics['f1']:.4f}")
    print()
    print("CANDIDATE")
    print(f"ROC-AUC   : {candidate_metrics['roc_auc']:.4f}")
    print(f"Precision : {candidate_metrics['precision']:.4f}")
    print(f"Recall    : {candidate_metrics['recall']:.4f}")
    print(f"F1        : {candidate_metrics['f1']:.4f}")
    print()
    print(f"ROC-AUC DELTA       : {delta:+.4f}")
    print(
        "CANDIDATE BETTER    :",
        "YES" if delta > 0 else "NO",
    )
    print(
        "PROMOTION RECOMMEND:",
        "YES" if delta > 0 else "NO",
    )
    print("HOLDOUT USED IN TRAINING: NO")
    print(f"RESULT: {OUTPUT}")


if __name__ == "__main__":
    run_experiment()
