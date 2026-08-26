import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score

from evaluation.redteam_false_positive import (
    make_benign_stress_set,
    make_true_fraud_set,
)

from evaluation.redteam_geo_curriculum import (
    build_curriculum,
    train_model,
)

from generation.generator import FEATURES


print("=== STAGE 8: FALSE-POSITIVE THRESHOLD SWEEP ===")

training = pd.concat(
    build_curriculum(seed=42),
    ignore_index=True,
)

model = train_model(training)

benign = make_benign_stress_set(
    n=10000,
    seed=4242,
)

fraud = make_true_fraud_set(
    n=10000,
    seed=5151,
)

mixed = pd.concat(
    [benign, fraud],
    ignore_index=True,
)

probabilities = model.predict_proba(
    mixed[FEATURES]
)[:, 1]

y = mixed["is_fraud"]

print()
print("THRESHOLD | BENIGN FPR | PRECISION | RECALL | F1")
print("-" * 58)

for threshold in np.arange(0.05, 0.96, 0.05):

    pred = (
        probabilities >= threshold
    ).astype(int)

    benign_pred = pred[:len(benign)]

    fpr = benign_pred.mean()

    precision = precision_score(
        y,
        pred,
        zero_division=0,
    )

    recall = recall_score(
        y,
        pred,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        pred,
        zero_division=0,
    )

    print(
        f"{threshold:9.2f} | "
        f"{fpr:10.4f} | "
        f"{precision:9.4f} | "
        f"{recall:6.4f} | "
        f"{f1:6.4f}"
    )

print()
print("Diagnostic only — no threshold is promoted.")
