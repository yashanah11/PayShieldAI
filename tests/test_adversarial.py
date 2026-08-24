from generation.fraud_injector import generate_fraud_dataset
from attacks.adversarial import generate_adversarial_cases


def test_adversarial_generation():
    df = generate_fraud_dataset(1000)
    attacked = generate_adversarial_cases(df)

    assert len(attacked) == len(df)
    assert "attack_type" in attacked.columns
    assert attacked["attack_type"].eq("adversarial_perturbation").all()
