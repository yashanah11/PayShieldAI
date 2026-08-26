import joblib
import numpy as np

from sklearn.metrics import roc_auc_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from generation.generator import FEATURES
from evaluation.hard_benchmark import generate_hard_benchmark


def train_candidate(seed=42):
    df = generate_hard_benchmark(
        n=10000,
        fraud_rate=0.05,
        seed=seed,
    )

    X_train, _, y_train, _ = train_test_split(
        df[FEATURES],
        df["is_fraud"],
        test_size=0.30,
        random_state=seed,
        stratify=df["is_fraud"],
    )

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

        df.loc[idx, "velocity_1h"] += rng.integers(
            2, 5, len(idx)
        )
        df.loc[idx, "velocity_24h"] += rng.integers(
            3, 10, len(idx)
        )

    elif family == "geographic":
        idx = fraud_indices

        df.loc[idx, "distance_km"] += rng.uniform(
            30, 120, len(idx)
        )

    elif family == "coordinated":
        idx = fraud_indices

        df.loc[idx, "amount"] *= rng.uniform(
            1.5, 3.0, len(idx)
        )

        df.loc[idx, "velocity_24h"] += rng.integers(
            2, 8, len(idx)
        )

        df.loc[idx, "distance_km"] += rng.uniform(
            20, 100, len(idx)
        )

    return df


def evaluate(model, df):
    X = df[FEATURES]
    y = df["is_fraud"]

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "auc": roc_auc_score(y, probabilities),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0,
        ),
    }


if __name__ == "__main__":
    print("=== RED-TEAM STAGE 3: ATTACK FAMILY DIAGNOSIS ===")

    model = train_candidate()

    base = generate_hard_benchmark(
        n=5000,
        fraud_rate=0.05,
        seed=999,
    )

    families = [
        "velocity",
        "geographic",
        "coordinated",
    ]

    results = {}

    for family in families:
        df = make_attack_family(
            base,
            family,
        )

        metrics = evaluate(
            model,
            df,
        )

        results[family] = metrics

        print()
        print(f"ATTACK FAMILY: {family.upper()}")
        print(f"ROC-AUC : {metrics['auc']:.4f}")
        print(f"Recall  : {metrics['recall']:.4f}")

    print()
    print("=== DIAGNOSIS ===")

    weakest = min(
        results,
        key=lambda x: results[x]["auc"],
    )

    print(f"Weakest family: {weakest}")

    if results[weakest]["auc"] < 0.80:
        print("STATUS: WEAKNESS CONFIRMED")
    else:
        print("STATUS: ALL FAMILIES PASS")
