from generation.generator import generate_transactions


def inject_fraud(df, fraud_rate=0.05, seed=42):
    result = df.copy()
    rng = __import__("numpy").random.default_rng(seed)

    n_fraud = int(len(result) * fraud_rate)

    if n_fraud == 0:
        return result

    indices = rng.choice(result.index, size=n_fraud, replace=False)

    result.loc[indices, "velocity_1h"] += rng.integers(3, 10, n_fraud)
    result.loc[indices, "velocity_24h"] += rng.integers(5, 20, n_fraud)
    result.loc[indices, "device_age_days"] = rng.integers(1, 30, n_fraud)
    result.loc[indices, "distance_km"] += rng.uniform(50, 500, n_fraud)
    result.loc[indices, "merchant_risk"] = rng.uniform(0.7, 1.0, n_fraud)
    result.loc[indices, "is_fraud"] = 1

    return result


def generate_fraud_dataset(n=10000, fraud_rate=0.05, seed=42):
    df = generate_transactions(n, seed)
    return inject_fraud(df, fraud_rate, seed)
