import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from generation.fraud_injector import generate_fraud_dataset
from generation.generator import FEATURES


MODEL_PATH = "models/xgboost_detector.joblib"


def find_hard_examples(df, model, threshold=0.5):
    probabilities = model.predict_proba(df[FEATURES])[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    missed = (df["is_fraud"].values == 1) & (predictions == 0)

    hard_examples = df.loc[missed].copy()
    hard_examples["model_probability"] = probabilities[missed]

    return hard_examples


def retrain_with_hard_examples(
    n=10000,
    seed=42,
    threshold=0.5,
):
    base = generate_fraud_dataset(
        n,
        fraud_rate=0.05,
        seed=seed,
    )

    old_model = joblib.load(MODEL_PATH)

    hard_examples = find_hard_examples(
        base,
        old_model,
        threshold,
    )

    # Combine original synthetic training data
    # with missed synthetic fraud cases.
    augmented = base.copy()

    if len(hard_examples) > 0:
        hard_clean = hard_examples[base.columns]
        augmented = np.concatenate(
            [augmented.to_records(index=False),
             hard_clean.to_records(index=False)]
        )

        import pandas as pd

        augmented = pd.DataFrame.from_records(
            augmented,
            columns=base.columns,
        )

    X = augmented[FEATURES]
    y = augmented["is_fraud"]

    from xgboost import XGBClassifier

    new_model = XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=seed,
    )

    new_model.fit(X, y)

    validation = generate_fraud_dataset(
        5000,
        fraud_rate=0.05,
        seed=seed + 1000,
    )

    old_prob = old_model.predict_proba(
        validation[FEATURES]
    )[:, 1]

    new_prob = new_model.predict_proba(
        validation[FEATURES]
    )[:, 1]

    old_auc = roc_auc_score(
        validation["is_fraud"],
        old_prob,
    )

    new_auc = roc_auc_score(
        validation["is_fraud"],
        new_prob,
    )

    result = {
        "training_transactions": int(len(base)),
        "hard_examples_found": int(len(hard_examples)),
        "augmented_training_size": int(len(augmented)),
        "old_roc_auc": float(old_auc),
        "new_roc_auc": float(new_auc),
        "improved": bool(new_auc >= old_auc),
    }

    Path("evaluation").mkdir(exist_ok=True)

    with open(
        "evaluation/self_improvement.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)

    if new_auc >= old_auc:
        joblib.dump(
            new_model,
            MODEL_PATH,
        )
        print("NEW MODEL ACCEPTED")
    else:
        print("OLD MODEL RETAINED")

    print("SELF-IMPROVEMENT: OK")
    print("HARD EXAMPLES:", len(hard_examples))
    print(f"OLD ROC-AUC: {old_auc:.4f}")
    print(f"NEW ROC-AUC: {new_auc:.4f}")
    print("IMPROVED:", new_auc >= old_auc)


if __name__ == "__main__":
    retrain_with_hard_examples()
