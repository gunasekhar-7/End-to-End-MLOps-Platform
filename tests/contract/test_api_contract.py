"""API Contract tests verifying OpenAPI schema adherence and stability."""

import pytest
from fastapi.testclient import TestClient

from mlops_fraud_detection.serve.main import (
    BatchPredictInput,
    BatchPredictOutput,
    HealthOutput,
    ModelInfoOutput,
    PredictionOutput,
    TransactionInput,
    app,
)


@pytest.mark.contract
class TestAPIContract:
    """Validate OpenAPI schema and JSON serialization contracts."""

    def test_openapi_schema_available(self):
        client = TestClient(app)
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "paths" in schema
        assert "/predict" in schema["paths"]
        assert "/predict/batch" in schema["paths"]
        assert "/health" in schema["paths"]
        assert "/model/info" in schema["paths"]

    def test_predict_endpoint_contract(self):
        client = TestClient(app)
        response = client.get("/openapi.json")
        schema = response.json()
        predict_spec = schema["paths"]["/predict"]["post"]

        assert "200" in predict_spec["responses"]
        assert "application/json" in predict_spec["responses"]["200"]["content"]
        assert "Predict" in predict_spec.get("summary", "") or "prediction" in predict_spec.get("description", "")

    def test_transaction_input_schema_contract(self):
        schema = TransactionInput.model_json_schema()
        assert "properties" in schema
        assert "amount" in schema["properties"]
        assert schema["properties"]["amount"]["type"] == "number"

    def test_prediction_output_schema_contract(self):
        schema = PredictionOutput.model_json_schema()
        assert "properties" in schema
        assert "fraud_probability" in schema["properties"]
        assert "is_fraud" in schema["properties"]
        assert "threshold_used" in schema["properties"]
        assert schema["required"] == ["fraud_probability", "is_fraud", "threshold_used"]

    def test_batch_predict_contract(self):
        input_schema = BatchPredictInput.model_json_schema()
        assert "transactions" in input_schema["properties"]

        schema = BatchPredictOutput.model_json_schema()
        assert "predictions" in schema["properties"]
        assert "summary" in schema["properties"]

    def test_health_output_contract(self):
        schema = HealthOutput.model_json_schema()
        assert set(schema["required"]) == {"status", "model_loaded", "uptime_seconds", "version"}

    def test_model_info_contract(self):
        schema = ModelInfoOutput.model_json_schema()
        assert "model_type" in schema["properties"]
        assert "threshold" in schema["properties"]
        assert "feature_count" in schema["properties"]
