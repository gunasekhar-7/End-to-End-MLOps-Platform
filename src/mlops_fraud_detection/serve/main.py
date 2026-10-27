"""FastAPI inference service for fraud detection."""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

import mlflow
from mlops_fraud_detection.common.utils import get_logger, load_params, setup_logging

# Setup logging
setup_logging("INFO")
logger = get_logger(__name__)

# Global model and preprocessor
model = None
preprocessor = None
model_info = {}
config = None


class TransactionInput(BaseModel):
    """Single transaction input for prediction."""

    model_config = ConfigDict(extra="allow")  # Allow extra fields for flexibility

    # Required fields - these should match training features
    amount: float = Field(..., description="Transaction amount")
    # Add other required features here based on your data

    # Common fraud detection features (optional, will use defaults if missing)
    time_since_last_tx: float | None = Field(
        None, description="Time since last transaction (hours)"
    )
    merchant_category: str | None = Field(None, description="Merchant category")
    tx_type: str | None = Field(None, description="Transaction type")
    country: str | None = Field(None, description="Country code")
    device_type: str | None = Field(None, description="Device type")
    user_id: str | None = Field(None, description="User identifier")
    card_present: bool | None = Field(None, description="Card present flag")


class BatchPredictInput(BaseModel):
    """Batch prediction input."""

    transactions: list[TransactionInput] = Field(..., min_length=1, max_length=10000)


class PredictionOutput(BaseModel):
    """Single prediction output."""

    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    is_fraud: bool = Field(..., description="Binary fraud prediction")
    threshold_used: float = Field(..., description="Classification threshold used")


class BatchPredictOutput(BaseModel):
    """Batch prediction output."""

    predictions: list[PredictionOutput]
    summary: dict[str, Any]


class ModelInfoOutput(BaseModel):
    """Model information output."""

    model_type: str
    model_version: str
    training_date: str
    git_sha: str
    metrics: dict[str, float]
    threshold: float
    feature_count: int


class HealthOutput(BaseModel):
    """Health check output."""

    status: str
    model_loaded: bool
    uptime_seconds: float
    version: str


# Track startup time
start_time = time.time()


def load_model_and_preprocessor():
    """Load model and preprocessor from disk or MLflow."""
    global model, preprocessor, model_info, config

    config = load_params()

    model_path = os.getenv("MODEL_PATH", config.model.model_file)
    preprocessor_path = os.getenv("PREPROCESSOR_PATH", config.model.preprocessor_file)
    mlflow_model_uri = os.getenv(
        "MLFLOW_MODEL_URI", config.mlflow.model_uri if hasattr(config.mlflow, "model_uri") else None
    )

    logger.info(
        "Loading model and preprocessor", model_path=model_path, preprocessor_path=preprocessor_path
    )

    # Try loading from MLflow Model Registry first
    if mlflow_model_uri:
        try:
            logger.info("Attempting to load model from MLflow", uri=mlflow_model_uri)
            model = mlflow.sklearn.load_model(mlflow_model_uri)
            # For preprocessor, we still need local file or MLflow artifact
            preprocessor = joblib.load(preprocessor_path)
            model_info["source"] = "mlflow_registry"
            model_info["mlflow_uri"] = mlflow_model_uri
            logger.info("Model loaded from MLflow Model Registry")
            return
        except Exception as e:
            logger.warning("Failed to load from MLflow, falling back to local", error=str(e))

    # Fallback to local files
    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        model_info["source"] = "local"
        logger.info("Model loaded from local files")
    except FileNotFoundError as e:
        logger.error("Model or preprocessor not found", error=str(e))
        raise RuntimeError(f"Model files not found: {e}")
    except Exception as e:
        logger.error("Failed to load model", error=str(e))
        raise RuntimeError(f"Failed to load model: {e}")

    # Load model metadata
    try:
        metrics = joblib.load("metrics/train.json") if Path("metrics/train.json").exists() else {}
        model_info.update(
            {
                "model_type": config.training.model_type,
                "model_version": "local",
                "training_date": "unknown",
                "git_sha": "unknown",
                "metrics": metrics,
                "threshold": config.serving.threshold,
                "feature_count": len(preprocessor.get_feature_names_out())
                if hasattr(preprocessor, "get_feature_names_out")
                else 0,
            }
        )
    except Exception as e:
        logger.warning("Could not load model metadata", error=str(e))
        model_info.update(
            {
                "model_type": config.training.model_type,
                "model_version": "local",
                "training_date": "unknown",
                "git_sha": "unknown",
                "metrics": {},
                "threshold": config.serving.threshold,
                "feature_count": 0,
            }
        )


def preprocess_input(input_data: dict[str, Any]) -> np.ndarray:
    """Preprocess input data for prediction."""
    global preprocessor, config

    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Ensure all expected columns exist (fill missing with defaults)
    if hasattr(preprocessor, "feature_names_in_"):
        expected_features = preprocessor.feature_names_in_
        for feat in expected_features:
            if feat not in df.columns:
                df[feat] = 0  # or appropriate default
        df = df[expected_features]

    # Transform
    try:
        processed = preprocessor.transform(df)
        return processed
    except Exception as e:
        logger.error("Preprocessing failed", error=str(e), input_keys=list(input_data.keys()))
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e!s}")


def predict_single(transaction: TransactionInput) -> PredictionOutput:
    """Make prediction for a single transaction."""
    global model, config

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Convert to dict
    input_dict = transaction.model_dump(exclude_none=True)

    # Preprocess
    processed = preprocess_input(input_dict)

    # Predict
    try:
        fraud_prob = float(model.predict_proba(processed)[0, 1])
        is_fraud = fraud_prob >= config.serving.threshold

        return PredictionOutput(
            fraud_probability=fraud_prob,
            is_fraud=bool(is_fraud),
            threshold_used=config.serving.threshold,
        )
    except Exception as e:
        logger.error("Prediction failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e!s}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global start_time
    start_time = time.time()

    # Startup
    logger.info("Starting Fraud Detection API")
    try:
        load_model_and_preprocessor()
        logger.info("API startup complete")
    except Exception as e:
        logger.error("Failed to load model on startup", error=str(e))
        # Don't raise - let health endpoint report status

    yield

    # Shutdown
    logger.info("Shutting down Fraud Detection API")


# Create FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="MLOps Fraud Detection Model Serving",
    version="0.1.0",
    lifespan=lifespan,
)

# Add Prometheus instrumentation
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/metrics", "/health"],
    inprogress_name="fraud_detection_inprogress",
    inprogress_labels=True,
)
instrumentator.instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error("Unhandled exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


@app.get("/health", response_model=HealthOutput)
async def health_check():
    """Health check endpoint."""
    return HealthOutput(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        uptime_seconds=time.time() - start_time,
        version="0.1.0",
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(transaction: TransactionInput):
    """Single transaction fraud prediction."""
    return predict_single(transaction)


@app.post("/predict/batch", response_model=BatchPredictOutput)
async def predict_batch(batch: BatchPredictInput):
    """Batch fraud prediction."""
    global model, config

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    predictions = []
    fraud_count = 0
    total_prob = 0.0

    for tx in batch.transactions:
        pred = predict_single(tx)
        predictions.append(pred)
        if pred.is_fraud:
            fraud_count += 1
        total_prob += pred.fraud_probability

    summary = {
        "total_transactions": len(predictions),
        "fraud_detected": fraud_count,
        "fraud_rate": fraud_count / len(predictions) if predictions else 0,
        "avg_fraud_probability": total_prob / len(predictions) if predictions else 0,
    }

    return BatchPredictOutput(predictions=predictions, summary=summary)


@app.get("/model/info", response_model=ModelInfoOutput)
async def model_info_endpoint():
    """Get model information."""
    if not model_info:
        raise HTTPException(status_code=503, detail="Model info not available")

return ModelInfoOutput(**model_info)


if __name__ == "__main__":
    import uvicorn

    config = load_params()
    uvicorn.run(
        "mlops_fraud_detection.serve.main:app",
        host=config.serving.host,
        port=config.serving.port,
        workers=config.serving.workers,
        log_level=config.serving.log_level.lower(),
        reload=False,
    )
