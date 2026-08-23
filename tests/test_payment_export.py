from payment.synthetic_payments import export_dataset


def test_payment_export():
    df = export_dataset(1000)

    assert len(df) == 1000
    assert "transaction_id" not in df.columns
    assert "is_fraud" in df.columns
