import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, recall_score
from evaluation.redteam_geo_curriculum import build_curriculum, train_model
from generation.generator import generate_transactions, FEATURES
from generation.hard_negatives import generate_legitimate_travelers


SEEDS = [7, 21, 42, 99, 123]
THRESHOLD = 0.10


def make_stealth_attacks(n=12000, fraud_rate=0.05, seed=42):
    rng = np.random.default_rng(seed)

    df = generate_transactions(n, seed)
    df["is_fraud"] = 0

    fraud_count = int(n * fraud_rate)

    indices = rng.choice(
        df.index,
        size=fraud_count,
        replace=False,
    )

    # Small coordinated changes across multiple features.
    df.loc[indices, "amount"] *= rng.uniform(1.3, 2.2, fraud_count)

    df.loc[indices, "velocity_1h"] += rng.integers(
        1, 4, fraud_count
    )

    df.loc[indices, "velocity_24h"] += rng.integers(
        2, 7, fraud_count
    )

    df.loc[indices, "distance_km"] += rng.uniform(
        20, 90, fraud_count
    )

    # Keep merchant and device signals deliberately normal.
    df.loc[indices, "merchant_risk"] = rng.uniform(
        0.15, 0.60, fraud_count
    )

    df.loc[indices, "device_age_days"] = rng.integers(
        100, 1200, fraud_count
    )

    df.loc[indices, "is_fraud"] = 1

    return df


print("=== STAGE 13: MULTI-FACTOR STEALTH ATTACK ===")

rows = []

for seed in SEEDS:

    training = pd.concat(
        build_curriculum(seed=seed),
        ignore_index=True,
    )

    model = train_model(training, seed=seed)

    fraud = make_stealth_attacks(
        n=12000,
        fraud_rate=0.05,
        seed=seed + 5000,
    )

    benign = generate_legitimate_travelers(
        n=12000,
        seed=seed + 6000,
    )

    fraud_prob = model.predict_proba(
        fraud[FEATURES]
    )[:, 1]

    benign_prob = model.predict_proba(
        benign[FEATURES]
    )[:, 1]

    fraud_pred = (
        fraud_prob >= THRESHOLD
    ).astype(int)

    auc = roc_auc_score(
        fraud["is_fraud"],
        fraud_prob,
    )

    recall = recall_score(
        fraud["is_fraud"],
        fraud_pred,
        zero_division=0,
    )

    fpr = float(
        (benign_prob >= THRESHOLD).mean()
    )

    rows.append((seed, auc, recall, fpr))

    print(
        f"Seed {seed:3d} | "
        f"AUC {auc:.4f} | "
        f"Recall {recall:.4f} | "
        f"FPR {fpr:.4f}"
    )


min_auc = min(x[1] for x in rows)
min_recall = min(x[2] for x in rows)
max_fpr = max(x[3] for x in rows)

print()
print(f"Minimum AUC    : {min_auc:.4f}")
print(f"Minimum recall : {min_recall:.4f}")
print(f"Maximum FPR    : {max_fpr:.4f}")

print()

if (
    min_auc >= 0.90
    and min_recall >= 0.70
    and max_fpr <= 0.05
):
    print("VERDICT: PASS")
else:
    print("VERDICT: FAIL")
