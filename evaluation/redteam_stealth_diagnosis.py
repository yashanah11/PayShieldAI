import numpy as np
import pandas as pd

from evaluation.redteam_stealth_attacks import make_stealth_attacks
from evaluation.redteam_geo_curriculum import build_curriculum, train_model
from generation.hard_negatives import generate_legitimate_travelers
from generation.generator import FEATURES


SEED = 42
THRESHOLD = 0.10

print("=== STAGE 13A: STEALTH FAILURE DIAGNOSIS ===")

training = pd.concat(
    build_curriculum(seed=SEED),
    ignore_index=True,
)

model = train_model(
    training,
    seed=SEED,
)

fraud = make_stealth_attacks(
    n=12000,
    fraud_rate=0.05,
    seed=SEED + 5000,
)

benign = generate_legitimate_travelers(
    n=12000,
    seed=SEED + 6000,
)

fraud_prob = model.predict_proba(
    fraud[FEATURES]
)[:, 1]

benign_prob = model.predict_proba(
    benign[FEATURES]
)[:, 1]

print()
print("=== FRAUD SCORE DISTRIBUTION ===")
print(
    pd.Series(fraud_prob).describe(
        percentiles=[
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
        ]
    )
)

print()
print("=== BENIGN SCORE DISTRIBUTION ===")
print(
    pd.Series(benign_prob).describe(
        percentiles=[
            0.90,
            0.95,
            0.99,
            0.995,
        ]
    )
)

print()
print("=== FRAUD BELOW THRESHOLD ===")

for threshold in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:

    missed = float(
        (fraud_prob < threshold).mean()
    )

    detected = 1.0 - missed

    benign_fpr = float(
        (benign_prob >= threshold).mean()
    )

    print(
        f"Threshold {threshold:.2f} | "
        f"Recall {detected:.4f} | "
        f"Benign FPR {benign_fpr:.4f}"
    )

print()
print("=== MISSED STEALTH ATTACKS ===")

missed_mask = fraud_prob < THRESHOLD

missed = fraud.loc[
    missed_mask,
    FEATURES,
]

detected = fraud.loc[
    ~missed_mask,
    FEATURES,
]

print(
    f"Missed attacks   : {len(missed)}"
)

print(
    f"Detected attacks : {len(detected)}"
)

print()
print("MISSED MEANS")
print(
    missed.mean(numeric_only=True)
)

print()
print("DETECTED MEANS")
print(
    detected.mean(numeric_only=True)
)

print()
print("=== FEATURE DIFFERENCE ===")

difference = (
    detected.mean(numeric_only=True)
    - missed.mean(numeric_only=True)
)

print(
    difference.sort_values(
        key=lambda x: abs(x),
        ascending=False,
    )
)

print()
print("=== DIAGNOSIS COMPLETE ===")
