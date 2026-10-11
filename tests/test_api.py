import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from fastapi.testclient import TestClient
from app.backend.main import app, load_artifacts

@pytest.fixture(scope="session", autouse=True)
def init_app():
    load_artifacts()

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["version"] == "1.0.0"

def test_predict_high_risk():
    payload = {
        "Amount": 31973.22,
        "Payment_type": "Debit card",
        "Payment_currency": "UK pounds",
        "Received_currency": "UK pounds",
        "Sender_bank_location": "UK",
        "Receiver_bank_location": "UK",
        "hour": 7,
        "day_of_week": 4,
        "is_weekend": 0,
        "sender_prior_count": 5,
        "sender_prior_mean": 170162.18,
        "sender_prior_std": 338656.62,
        "amount_z_score": -0.408,
        "time_since_prev_sender_txn": 167.1,
        "receiver_prior_in_count": 14,
        "sender_receiver_prior_count": 0,
        "distinct_prior_receivers": 4
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "calibrated_prob" in data
    assert data["risk_tier"] in ["Critical", "High", "Medium"]
    assert "decision" in data
    assert data["expected_loss"] > 0
    assert len(data["top_3_reasons"]) == 3

def test_predict_low_risk():
    payload = {
        "Amount": 45.50,
        "Payment_type": "Debit card",
        "Payment_currency": "UK pounds",
        "Received_currency": "UK pounds",
        "Sender_bank_location": "UK",
        "Receiver_bank_location": "UK",
        "hour": 14,
        "day_of_week": 2,
        "is_weekend": 0,
        "sender_prior_count": 50,
        "sender_prior_mean": 55.0,
        "sender_prior_std": 20.0,
        "amount_z_score": -0.4,
        "time_since_prev_sender_txn": 72.0,
        "receiver_prior_in_count": 120,
        "sender_receiver_prior_count": 15,
        "distinct_prior_receivers": 10
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_tier"] == "Low"
    assert data["decision"] == "Automated Approval"
    assert len(data["top_3_reasons"]) == 3

def test_predict_batch():
    csv_content = (
        "Amount,Payment_type,Payment_currency,Received_currency,Sender_bank_location,Receiver_bank_location\n"
        "15000.0,Cross-border,US dollar,Euro,UK,Germany\n"
        "25.0,Debit card,UK pounds,UK pounds,UK,UK\n"
    )
    files = {"file": ("test.csv", csv_content.encode("utf-8"), "text/csv")}
    response = client.post("/predict_batch", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["total_scored"] == 2
    assert "csv_data" in data

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "business_summary" in data

def test_bi_simulate():
    payload = {
        "capacity_pct": 1.0,
        "review_cost_usd": 50.0
    }
    response = client.post("/bi_simulate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["budget_pct"] == 1.0
    assert data["projected_recall"] > 0.90
    assert data["net_financial_savings_usd"] > 0
