import numpy as np
import pandas as pd


def perturb_transactions(df, seed=42):
    result = df.copy()
    rng = np.random.default_rng(seed)

    n = len(result)
    indices = rng.choice(result.index, size=max(1, n // 20), replace=False)

    result.loc[indices, "amount"] *= rng.uniform(0.85, 1.15, len(indices))
    result.loc[indices, "velocity_1h"] = np.maximum(
        0, result.loc[indices, "velocity_1h"] - rng.integers(1, 3, len(indices))
    )
    result.loc[indices, "merchant_risk"] *= rng.uniform(
        0.7, 0.95, len(indices)
    )

    result["attack_type"] = "adversarial_perturbation"

    return result


def generate_adversarial_cases(df, seed=42):
    return perturb_transactions(df, seed)
