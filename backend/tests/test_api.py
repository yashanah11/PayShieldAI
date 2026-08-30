import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

valid_txn = {
    "amount": 100.0,
    "hour": 12.0,
    "velocity_1h": 2.0,
    "velocity_24h": 5.0,
    "device_age_days": 200.0,
    "distance_km": 10.0,
    "merchant_risk": 0.05
}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict():
    response = client.post("/predict", json=valid_txn)
    assert response.status_code == 200
    assert "fraud_probability" in response.json()

def test_get_attacks():
    response = client.get("/attacks")
    assert response.status_code == 200
    assert len(response.json()["attack_families"]) == 8

def test_simulate_attack():
    payload = {
        "transaction": valid_txn,
        "attack_family": "Velocity Spike"
    }
    response = client.post("/simulate-attack", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "velocity_1h" in data["changed_features"]
    assert "velocity_24h" in data["changed_features"]

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["model_version"] == "v6_robust"