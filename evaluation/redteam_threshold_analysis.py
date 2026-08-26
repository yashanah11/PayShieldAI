import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

from evaluation.redteam_geo_curriculum import (
    make_geo_dataset,
    build_curriculum,
    train_model,
)

import pandas as pd

print("=== STAGE 5: GEOGRAPHIC THRESHOLD ANALYSIS ===")

curriculum = build_curriculum()

training = pd.concat(
    curriculum,
    ignore_index=True,
)

model = train_model(training)

holdout = make_geo_dataset(
    n=10000,
    fraud_rate=0.05,
    seed=9999,
    level=5,
)

probabilities = model.predict_proba(
    holdout[[
        "amount",
        "hour",
        "velocity_1h",
        "velocity_24h",
        "device_age_days",
        "distance_km",
        "merchant_risk",
    ]]
)[:, 1]

y = holdout["is_fraud"]

auc = roc_auc_score(y, probabilities)

print(f"ROC-AUC: {auc:.4f}")
print()

print(
    "THRESHOLD | PRECISION | RECALL | F1 | FRAUD FLAGS"
)

print("-" * 55)

for threshold in np.arange(0.05, 0.96, 0.05):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    flags = int(predictions.sum())

    print(
        f"{threshold:8.2f} | "
        f"{precision:9.4f} | "
        f"{recall:6.4f} | "
        f"{f1:6.4f} | "
        f"{flags:11d}"
    )

print()
print("=== INTERPRETATION ===")
print(
    "AUC measures ranking quality; threshold determines "
    "the final fraud decision."
)
