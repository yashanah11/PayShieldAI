import numpy as np
import pandas as pd

from evaluation.redteam_stealth_curriculum import (
    build_stealth_curriculum,
    train_model,
)

from generation.hard_negatives import generate_legitimate_travelers
from generation.generator import FEATURES


SEED = 42
THRESHOLD = 0.10

print("=== STAGE 14B: STEALTH FALSE-POSITIVE DIAGNOSIS ===")

training = pd.concat(
    build_stealth_curriculum(seed=SEED),
    ignore_index=True,
)

model = train_model(
    training,
    seed=SEED,
)

benign = generate_legitimate_travelers(
    n=20000,
    seed=424242,
)

scores = model.predict_proba(
    benign[FEATURES]
)[:, 1]

benign = benign.copy()
benign["fraud_score"] = scores
benign["predicted_fraud"] = (
    scores >= THRESHOLD
).astype(int)

print()
print("=== SCORE DISTRIBUTION ===")

print(
    benign["fraud_score"].describe(
        percentiles=[
            0.90,
            0.95,
            0.97,
            0.98,
            0.99,
            0.995,
            0.999,
        ]
    )
)

print()
print("=== FALSE POSITIVES ===")

fp = benign[
    benign["predicted_fraud"] == 1
]

print(
    f"Benign transactions : {len(benign)}"
)

print(
    f"False positives      : {len(fp)}"
)

print(
    f"False-positive rate  : "
    f"{len(fp) / len(benign):.4f}"
)

print()
print("=== FEATURE MEANS ===")

print("ALL BENIGN")

print(
    benign[FEATURES].mean()
)

print()

print("FALSE POSITIVES")

print(
    fp[FEATURES].mean()
)

print()

print("=== FEATURE DIFFERENCE ===")

difference = (
    fp[FEATURES].mean()
    - benign[FEATURES].mean()
)

print(
    difference.sort_values(
        key=np.abs,
        ascending=False,
    )
)

print()
print("=== TOP 20 FALSE POSITIVES ===")

print(
    fp.sort_values(
        "fraud_score",
        ascending=False,
    )[FEATURES + ["fraud_score"]]
    .head(20)
    .to_string(index=False)
)

print()
print("=== FEATURE IMPORTANCE ===")

importance = pd.Series(
    model.feature_importances_,
    index=FEATURES,
)

print(
    importance.sort_values(
        ascending=False
    )
)

print()
print("=== DIAGNOSIS COMPLETE ===")
