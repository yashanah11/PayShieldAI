"""
Stage 10: Adversarial Dataset Generation
PayShieldAI System

Creates a non-destructive adversarial training dataset by injecting
the 8 locked red-team attack families into clean baseline traffic.

IMPORTANT:
- Does NOT modify generation/generator.py
- Does NOT modify v1/v3
- Does NOT overwrite existing models
- Uses only the canonical 7-feature schema
"""

import os
import sys
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generation.generator import generate_transactions
from evaluation.redteam_attack_families import get_redteam_attacks


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

DATA_DIR = os.path.join(PROJECT_ROOT, "data")

OUTPUT_PATH = os.path.join(
    DATA_DIR,
    "stage10_adversarial_train.csv"
)


# ------------------------------------------------------------
# LOCKED FEATURE SCHEMA
# ------------------------------------------------------------

LOCKED_FEATURES = [
    "amount",
    "hour",
    "velocity_1h",
    "velocity_24h",
    "device_age_days",
    "distance_km",
    "merchant_risk",
]


# ------------------------------------------------------------
# DATASET GENERATION
# ------------------------------------------------------------

def create_adversarial_dataset(
    n_total=100000,
    fraud_ratio=0.10,
    seed=42,
):

    print("=" * 70)
    print("PAYSHIELD-AI: STAGE 10 ADVERSARIAL DATA GENERATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Validate configuration
    # --------------------------------------------------------

    if fraud_ratio <= 0 or fraud_ratio >= 1:
        raise ValueError(
            "fraud_ratio must be between 0 and 1."
        )

    # --------------------------------------------------------
    # Load locked attack families
    # --------------------------------------------------------

    attacks = get_redteam_attacks()

    if len(attacks) != 8:
        raise ValueError(
            f"CRITICAL: Expected exactly 8 locked attack families, "
            f"got {len(attacks)}."
        )

    print(
        f"[INFO] Verified exactly {len(attacks)} locked attack families."
    )

    # --------------------------------------------------------
    # Generate CLEAN baseline
    # --------------------------------------------------------

    print(
        f"[INFO] Generating {n_total} clean baseline transactions..."
    )

    base_df = generate_transactions(
        n=n_total,
        seed=seed,
    )

    # --------------------------------------------------------
    # Verify generator output
    # --------------------------------------------------------

    missing = [
        feature
        for feature in LOCKED_FEATURES
        if feature not in base_df.columns
    ]

    if missing:
        raise ValueError(
            f"CRITICAL: Generator missing features: {missing}"
        )

    if "is_fraud" not in base_df.columns:
        raise ValueError(
            "CRITICAL: Generator output missing is_fraud column."
        )

    generator_fraud = int(
        base_df["is_fraud"].sum()
    )

    print(
        f"[INFO] Generator fraud labels: {generator_fraud}"
    )

    if generator_fraud != 0:
        raise ValueError(
            "CRITICAL: Stage 10 expects the canonical generator "
            "to produce clean traffic only."
        )

    # --------------------------------------------------------
    # Calculate dataset sizes
    # --------------------------------------------------------

    n_fraud = int(
        n_total * fraud_ratio
    )

    n_benign = n_total - n_fraud

    print()
    print(
        f"[INFO] Total samples       : {n_total}"
    )

    print(
        f"[INFO] Target fraud samples: {n_fraud}"
    )

    print(
        f"[INFO] Benign samples      : {n_benign}"
    )

    # --------------------------------------------------------
    # Create deterministic attack allocation
    # --------------------------------------------------------

    n_attacks = len(attacks)

    samples_per_attack = n_fraud // n_attacks

    remainder = n_fraud % n_attacks

    print(
        f"[INFO] Attack families     : {n_attacks}"
    )

    print(
        f"[INFO] Base samples/attack : {samples_per_attack}"
    )

    # --------------------------------------------------------
    # Partition baseline
    #
    # First n_benign rows become benign traffic.
    # Remaining rows become attack source profiles.
    # --------------------------------------------------------

    X_benign = base_df.iloc[
        :n_benign
    ][LOCKED_FEATURES].copy()

    X_attack_pool = base_df.iloc[
        n_benign:
    ][LOCKED_FEATURES].copy()

    if len(X_attack_pool) != n_fraud:
        raise ValueError(
            "CRITICAL: Attack source pool size mismatch."
        )

    y_benign = np.zeros(
        n_benign,
        dtype=int
    )

    # --------------------------------------------------------
    # Inject attacks
    # --------------------------------------------------------

    attacked_dfs = []
    attacked_labels = []

    cursor = 0

    print()
    print("-" * 70)
    print("INJECTING LOCKED RED-TEAM ATTACK FAMILIES")
    print("-" * 70)

    attack_items = list(attacks.items())

    for attack_index, (
        attack_name,
        attack_fn
    ) in enumerate(attack_items):

        # Give the remainder to the final attack family.
        current_count = samples_per_attack

        if attack_index == n_attacks - 1:
            current_count += remainder

        start = cursor
        end = cursor + current_count

        X_target = X_attack_pool.iloc[
            start:end
        ].copy()

        y_target = np.ones(
            current_count,
            dtype=int
        )

        # Apply locked attack transformation
        X_adv, y_adv = attack_fn(
            X_target,
            y_target
        )

        # ----------------------------------------------------
        # Strict validation
        # ----------------------------------------------------

        if not isinstance(
            X_adv,
            pd.DataFrame
        ):
            X_adv = pd.DataFrame(
                X_adv,
                columns=LOCKED_FEATURES
            )

        missing_adv = [
            feature
            for feature in LOCKED_FEATURES
            if feature not in X_adv.columns
        ]

        if missing_adv:
            raise ValueError(
                f"Attack '{attack_name}' removed required "
                f"features: {missing_adv}"
            )

        X_adv = X_adv[
            LOCKED_FEATURES
        ].copy()

        y_adv = np.asarray(
            y_adv,
            dtype=int
        )

        if len(X_adv) != current_count:
            raise ValueError(
                f"Attack '{attack_name}' changed sample count."
            )

        if len(y_adv) != current_count:
            raise ValueError(
                f"Attack '{attack_name}' returned "
                f"incorrect label count."
            )

        if not np.all(y_adv == 1):
            raise ValueError(
                f"Attack '{attack_name}' did not preserve "
                f"fraud labels."
            )

        attacked_dfs.append(
            X_adv
        )

        attacked_labels.append(
            y_adv
        )

        print(
            f"[{attack_index + 1}/8] "
            f"{attack_name:<22} "
            f"Samples: {current_count}"
        )

        cursor = end

    # --------------------------------------------------------
    # Recombine
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("RECOMBINING DATASET")
    print("-" * 70)

    X_attacked = pd.concat(
        attacked_dfs,
        ignore_index=True
    )

    y_attacked = np.concatenate(
        attacked_labels
    )

    X_final = pd.concat(
        [
            X_benign,
            X_attacked
        ],
        ignore_index=True
    )

    y_final = np.concatenate(
        [
            y_benign,
            y_attacked
        ]
    )

    # --------------------------------------------------------
    # Final dataframe
    # --------------------------------------------------------

    final_df = X_final.copy()

    final_df["is_fraud"] = y_final

    # --------------------------------------------------------
    # Deterministic shuffle
    # --------------------------------------------------------

    final_df = final_df.sample(
        frac=1.0,
        random_state=seed
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    expected_columns = (
        LOCKED_FEATURES
        + ["is_fraud"]
    )

    if list(final_df.columns) != expected_columns:
        raise ValueError(
            "CRITICAL: Final dataset schema mismatch.\n"
            f"Expected: {expected_columns}\n"
            f"Got:      {list(final_df.columns)}"
        )

    if len(final_df) != n_total:
        raise ValueError(
            "CRITICAL: Final dataset row count mismatch."
        )

    actual_fraud = int(
        final_df["is_fraud"].sum()
    )

    if actual_fraud != n_fraud:
        raise ValueError(
            f"CRITICAL: Expected {n_fraud} fraud samples, "
            f"got {actual_fraud}."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        DATA_DIR,
        exist_ok=True
    )

    final_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STAGE 10 DATASET COMPLETE")
    print("=" * 70)

    print(
        f"[SUCCESS] Saved to:"
    )

    print(
        f"          {OUTPUT_PATH}"
    )

    print()
    print(
        f"Total rows       : {len(final_df)}"
    )

    print(
        f"Total columns    : {len(final_df.columns)}"
    )

    print(
        f"Benign samples   : "
        f"{int((final_df['is_fraud'] == 0).sum())}"
    )

    print(
        f"Fraud samples    : "
        f"{int((final_df['is_fraud'] == 1).sum())}"
    )

    print(
        f"Fraud prevalence : "
        f"{final_df['is_fraud'].mean() * 100:.2f}%"
    )

    print()
    print("Schema:")

    for feature in final_df.columns:
        print(
            f"  - {feature}"
        )

    print("=" * 70)


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":

    create_adversarial_dataset(
        n_total=100000,
        fraud_ratio=0.10,
        seed=42,
    )