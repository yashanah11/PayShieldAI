import json
from pathlib import Path

import pandas as pd

from generation.fraud_injector import generate_fraud_dataset


def to_payment_records(df):
    records = []

    for i, row in df.iterrows():
        records.append({
            "transaction_id": f"TXN-{i + 1:08d}",
            "rail": "UPI",
            "amount": float(row["amount"]),
            "hour": int(row["hour"]),
            "velocity_1h": int(row["velocity_1h"]),
            "velocity_24h": int(row["velocity_24h"]),
            "device_age_days": int(row["device_age_days"]),
            "distance_km": float(row["distance_km"]),
            "merchant_risk": float(row["merchant_risk"]),
            "is_fraud": int(row["is_fraud"]),
        })

    return records


def export_dataset(n=10000, seed=42):
    df = generate_fraud_dataset(n, fraud_rate=0.05, seed=seed)

    Path("data").mkdir(exist_ok=True)

    df.to_csv("data/synthetic_transactions.csv", index=False)

    records = to_payment_records(df)

    with open(
        "data/synthetic_payment_messages.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(records, f, indent=2)

    return df


if __name__ == "__main__":
    df = export_dataset()

    print("PAYMENT DATA: OK")
    print("TRANSACTIONS:", len(df))
    print("FRAUD:", int(df["is_fraud"].sum()))
    print("CSV: data/synthetic_transactions.csv")
    print("JSON: data/synthetic_payment_messages.json")
