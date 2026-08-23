from generation.fraud_injector import generate_fraud_dataset


def test_fraud_injection():
    df = generate_fraud_dataset(10000, fraud_rate=0.05)

    assert len(df) == 10000
    assert df["is_fraud"].sum() == 500
    assert set(df["is_fraud"].unique()) == {0, 1}
