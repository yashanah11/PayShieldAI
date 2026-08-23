import numpy as np
import pandas as pd

FEATURES = [
    "amount",
    "hour",
    "velocity_1h",
    "velocity_24h",
    "device_age_days",
    "distance_km",
    "merchant_risk",
]

def generate_transactions(n=1000, seed=42):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "amount": np.round(rng.lognormal(5.0, 1.0, n), 2),
        "hour": rng.integers(0, 24, n),
        "velocity_1h": rng.poisson(1.5, n),
        "velocity_24h": rng.poisson(5, n),
        "device_age_days": rng.integers(1, 1500, n),
        "distance_km": np.round(rng.exponential(25, n), 2),
        "merchant_risk": np.round(rng.beta(2, 5, n), 4),
    })

    df["is_fraud"] = 0
    return df
