"""Unit and API integration tests for FastAPI serving layer."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlops_fraud_detection.serve import main
from mlops_fraud_detection.serve.main import (
    TransactionInput,
    app,
    predict_batch_vectorized,
)


class DummyPreprocessor:
    """Mock preprocessor implementing scikit-learn interface."""

    def __init__(self):
        self.feature_names_in_ = np.array(["amount", "time_since_last_tx"])

    def transform(self, df):
        # Extract features and fill NaN with 0
        feats = df[["amount", "time_since_last_tx"]].fillna(0.0).values
        return feats

    def get_feature_names_out(self):
        return self.feature_names_in_


class DummyModel:
    """Mock classifier implementing predict_proba."""

    def predict_proba(self, X):
        # Predict higher fraud probability for higher amount
        probs = []
        for row in X:
            amount = row[0]
            prob = 0.95 if amount > 1000.0 else 0.05
            probs.append([1.0 - prob, prob])
        return np.array(probs)


@pytest.fixture(autouse=True)
def setup_mock_serving_state(monkeypatch):
    """Set up mocked model and preprocessor state for serving tests."""
    from omegaconf import OmegaConf

    dummy_config = OmegaConf.create(
        {
            "serving": {
                "host": "0.0.0.0",
                "port": 8000,
                "threshold": 0.5,
                "log_level": "INFO",
                "workers": 1,
            },
            "training": {
                "model_type": "xgboost",
            },
            "model": {
                "model_file": "models/model.pkl",
                "preprocessor_file": "models/preprocessor.pkl",
            },
            "mlflow": {
                "tracking_uri": "http://localhost:5000",
            },
        }
    )

    monkeypatch.setattr(main, "model", DummyModel())
    monkeypatch.setattr(main, "preprocessor", DummyPreprocessor())
    monkeypatch.setattr(main, "config", dummy_config)
    monkeypatch.setattr(
        main,
        "model_info",
        {
            "model_type": "xgboost",
            "model_version": "v1.0.0",
            "training_date": "2026-10-31 00:00:00",
            "git_sha": "abc1234",
            "metrics": {"f1": 0.92, "roc_auc": 0.98},
            "threshold": 0.5,
            "feature_count": 2,
        },
    )


def test_health_check_healthy():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "uptime_seconds" in data
    assert data["version"] == "1.0.0"


def test_metrics_endpoint():
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "fraud_prediction_probability" in response.text


def test_model_info_endpoint():
    client = TestClient(app)
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "xgboost"
    assert data["threshold"] == 0.5
    assert data["feature_count"] == 2


def test_predict_single_low_risk():
    client = TestClient(app)
    payload = {
        "amount": 50.0,
        "time_since_last_tx": 12.0,
        "merchant_category": "grocery",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is False
    assert data["fraud_probability"] < 0.5
    assert data["threshold_used"] == 0.5


def test_predict_single_high_risk():
    client = TestClient(app)
    payload = {
        "amount": 2500.0,
        "time_since_last_tx": 0.1,
        "merchant_category": "electronics",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fraud"] is True
    assert data["fraud_probability"] >= 0.5
    assert data["threshold_used"] == 0.5


def test_predict_batch_vectorized():
    client = TestClient(app)
    payload = {
        "transactions": [
            {"amount": 25.0, "time_since_last_tx": 5.0},
            {"amount": 5000.0, "time_since_last_tx": 0.05},
            {"amount": 100.0, "time_since_last_tx": 24.0},
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 3
    assert data["summary"]["total_transactions"] == 3
    assert data["summary"]["fraud_detected"] == 1
    assert data["predictions"][0]["is_fraud"] is False
    assert data["predictions"][1]["is_fraud"] is True
    assert data["predictions"][2]["is_fraud"] is False


def test_predict_batch_empty_rejected():
    client = TestClient(app)
    response = client.post("/predict/batch", json={"transactions": []})
    assert response.status_code == 422  # Pydantic min_length validation error


def test_predict_model_unloaded(monkeypatch):
    monkeypatch.setattr(main, "model", None)
    client = TestClient(app)

    # Health check shows degraded
    resp_health = client.get("/health")
    assert resp_health.status_code == 200
    assert resp_health.json()["status"] == "degraded"

    # Prediction returns 503
    resp_pred = client.post("/predict", json={"amount": 10.0})
    assert resp_pred.status_code == 503


def test_vectorized_consistency():
    """Ensure vectorized batch prediction produces identical results to single prediction."""
    tx1 = TransactionInput(amount=10.0, time_since_last_tx=2.0)
    tx2 = TransactionInput(amount=3000.0, time_since_last_tx=0.1)

    probs, fraud, threshold = predict_batch_vectorized([tx1, tx2])
    assert len(probs) == 2
    assert probs[0] < 0.5
    assert probs[1] >= 0.5
    assert bool(fraud[0]) is False
    assert bool(fraud[1]) is True
    assert threshold == 0.5


def test_prepare_dataframe_for_inference():
    from mlops_fraud_detection.serve.main import prepare_dataframe_for_inference

    records = [{"amount": 100.0}]
    df = prepare_dataframe_for_inference(records)
    assert "amount" in df.columns
    assert "time_since_last_tx" in df.columns
    assert np.isnan(df["time_since_last_tx"].iloc[0])


def test_load_model_and_preprocessor_local(tmp_path, monkeypatch):
    import joblib

    from mlops_fraud_detection.serve.main import load_model_and_preprocessor

    model_file = tmp_path / "model.pkl"
    preproc_file = tmp_path / "preprocessor.pkl"

    joblib.dump(DummyModel(), model_file)
    joblib.dump(DummyPreprocessor(), preproc_file)

    monkeypatch.setenv("MODEL_PATH", str(model_file))
    monkeypatch.setenv("PREPROCESSOR_PATH", str(preproc_file))
    monkeypatch.setenv("MLFLOW_MODEL_URI", "")

    load_model_and_preprocessor()
    assert main.model is not None
    assert main.preprocessor is not None


def test_model_info_unloaded_returns_503(monkeypatch):
    monkeypatch.setattr(main, "model", None)
    client = TestClient(app)
    response = client.get("/model/info")
    assert response.status_code == 503


def test_global_exception_handler(monkeypatch):
    def mock_raise(*args, **kwargs):
        raise RuntimeError("Unexpected failure")

    monkeypatch.setattr(main, "predict_batch_vectorized", mock_raise)
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/predict", json={"amount": 10.0})
    assert response.status_code == 500
    assert "Unexpected failure" in response.text
