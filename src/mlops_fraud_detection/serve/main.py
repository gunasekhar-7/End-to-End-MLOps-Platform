"""FastAPI inference service for fraud detection."""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, ConfigDict, Field

from mlops_fraud_detection.common.utils import get_logger, load_params, setup_logging

# Setup logging
setup_logging("INFO")
logger = get_logger(__name__)

# Global model, preprocessor, and configuration
model: Any = None
preprocessor: Any = None
model_info: dict[str, Any] = {}
config: Any = None

# Custom Prometheus Metrics for MLOps Observability & Alerting
FRAUD_PREDICTIONS_TOTAL = Counter(
    "fraud_predictions_total",
    "Total count of fraud inference predictions",
    ["is_fraud"],
)
FRAUD_PREDICTION_PROB_HISTOGRAM = Histogram(
    "fraud_prediction_probability_distribution",
    "Distribution of fraud prediction probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
FRAUD_PRED_PROB_GAUGE = Gauge(
    "fraud_prediction_probability",
    "Latest fraud prediction probability",
    ["endpoint"],
)
FRAUD_PRED_STDDEV_GAUGE = Gauge(
    "fraud_prediction_probability_stddev",
    "Standard deviation of recent prediction probabilities",
    ["endpoint"],
)
FRAUD_MODEL_LAST_TRAINED = Gauge(
    "fraud_model_last_trained_timestamp",
    "Epoch timestamp when current active model was trained",
)


class TransactionInput(BaseModel):
    """Single transaction input for prediction.

    Accepts domain features with safe defaults and permits extra features
    to support enterprise feature stores.
    """

    amount: float = Field(default=0.0, description="Transaction amount")
    time_since_last_tx: float | None = Field(
        None, description="Time since last transaction (hours)"
    )
    merchant_category: str | None = Field(
        None, description="Merchant category (e.g., grocery, online, travel)"
    )
    tx_type: str | None = Field(None, description="Transaction type (online, pos, atm)")
    country: str | None = Field(None, description="Country code (e.g., US, CA, UK, DE)")
    device_type: str | None = Field(None, description="Device type (e.g., mobile, desktop, tablet)")
    user_id: str | None = Field(None, description="User identifier")
    card_present: bool | None = Field(None, description="Whether card was physically present")

    model_config = ConfigDict(
        extra="allow",
    )


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


def load_model_and_preprocessor() -> None:
    """Load model and preprocessor from disk or MLflow."""
    global model, preprocessor, model_info, config

    config = load_params()

    model_path = os.getenv("MODEL_PATH", config.model.model_file)
    preprocessor_path = os.getenv("PREPROCESSOR_PATH", config.model.preprocessor_file)
    mlflow_model_uri = os.getenv(
        "MLFLOW_MODEL_URI",
        config.mlflow.model_uri if hasattr(config.mlflow, "model_uri") else None,
    )

    logger.info(
        "Loading model and preprocessor",
        model_path=model_path,
        preprocessor_path=preprocessor_path,
    )

    # Try loading from MLflow Model Registry first
    if mlflow_model_uri:
        try:
            logger.info("Attempting to load model from MLflow", uri=mlflow_model_uri)
            model = mlflow.sklearn.load_model(mlflow_model_uri)
            preprocessor = joblib.load(preprocessor_path)
            model_info["source"] = "mlflow_registry"
            model_info["mlflow_uri"] = mlflow_model_uri
            logger.info("Model loaded from MLflow Model Registry")
        except Exception as e:
            logger.warning("Failed to load from MLflow, falling back to local", error=str(e))

    # Fallback to local files
    if model is None:
        try:
            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)
            model_info["source"] = "local"
            logger.info("Model loaded from local files")
        except FileNotFoundError as e:
            logger.error("Model or preprocessor not found", error=str(e))
            raise RuntimeError(f"Model files not found: {e}") from None
        except Exception as e:
            logger.error("Failed to load model", error=str(e))
            raise RuntimeError(f"Failed to load model: {e}") from None

    # Determine training timestamp for Prometheus
    last_trained_ts = time.time()
    if Path(model_path).exists():
        last_trained_ts = Path(model_path).stat().st_mtime
    FRAUD_MODEL_LAST_TRAINED.set(last_trained_ts)

    # Load model metadata
    metrics_path = Path("metrics/train.json")
    metrics: dict[str, float] = {}
    git_sha = "unknown"
    if metrics_path.exists():
        try:
            import json

            with metrics_path.open() as f:
                loaded = json.load(f)
                git_sha = str(loaded.get("git_sha", "unknown"))
                metrics = {k: float(v) for k, v in loaded.items() if isinstance(v, (int, float))}
        except Exception as e:
            logger.warning("Failed to read metrics json", error=str(e))

    model_info.update(
        {
            "model_type": str(config.training.model_type),
            "model_version": "production",
            "training_date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_trained_ts)),
            "git_sha": git_sha,
            "metrics": metrics,
            "threshold": float(config.serving.threshold),
            "feature_count": len(preprocessor.get_feature_names_out())
            if hasattr(preprocessor, "get_feature_names_out")
            else (
                len(getattr(preprocessor, "feature_names_in_", []))
                if hasattr(preprocessor, "feature_names_in_")
                else 0
            ),
        }
    )


def prepare_dataframe_for_inference(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Vectorized dataframe preparation and feature alignment."""
    global preprocessor

    df = pd.DataFrame(records)

    # Align with preprocessor features if available
    if hasattr(preprocessor, "feature_names_in_"):
        expected_features = preprocessor.feature_names_in_

        # Map common aliases if missing
        if "TransactionAmt" in expected_features and "TransactionAmt" not in df.columns:
            if "amount" in df.columns:
                df["TransactionAmt"] = df["amount"]

        # Ensure all expected columns are present, using np.nan so imputer handles them correctly
        for feat in expected_features:
            if feat not in df.columns:
                df[feat] = np.nan

        df = df[expected_features]

    return df


def predict_batch_vectorized(
    transactions: list[TransactionInput],
) -> tuple[np.ndarray, np.ndarray, float]:
    """Execute high-performance vectorized inference over a batch."""
    global model, preprocessor, config

    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded")

    threshold = float(config.serving.threshold)

    # 1. Build single DataFrame from list of dicts
    records = [tx.model_dump(exclude_none=True) for tx in transactions]
    df = prepare_dataframe_for_inference(records)

    # 2. Vectorized transformation
    try:
        processed = preprocessor.transform(df)
    except Exception as e:
        logger.error("Preprocessing failed", error=str(e))
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e!s}") from None

    # 3. Vectorized model inference
    try:
        probabilities = model.predict_proba(processed)[:, 1]
        is_fraud = probabilities >= threshold
    except Exception as e:
        logger.error("Inference execution failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Inference failed: {e!s}") from None

    # 4. Update Prometheus metrics
    for prob, fraud in zip(probabilities, is_fraud, strict=False):
        FRAUD_PREDICTIONS_TOTAL.labels(is_fraud=str(bool(fraud)).lower()).inc()
        FRAUD_PREDICTION_PROB_HISTOGRAM.observe(float(prob))

    mean_prob = float(np.mean(probabilities))
    std_prob = float(np.std(probabilities)) if len(probabilities) > 1 else 0.05
    FRAUD_PRED_PROB_GAUGE.labels(endpoint="/predict").set(mean_prob)
    FRAUD_PRED_STDDEV_GAUGE.labels(endpoint="/predict").set(std_prob)

    return probabilities, is_fraud, threshold


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global start_time
    start_time = time.time()

    logger.info("Starting Fraud Detection API")
    try:
        load_model_and_preprocessor()
        logger.info("API startup complete")
    except Exception as e:
        logger.error("Failed to load model on startup", error=str(e))

    yield

    logger.info("Shutting down Fraud Detection API")


# Create FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Production-Grade MLOps Fraud Detection Inference Service",
    version="1.0.0",
    lifespan=lifespan,
)

# Add Prometheus instrumentation for HTTP stats
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
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    logger.error("Unhandled exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


@app.get("/health", response_model=HealthOutput)
async def health_check() -> HealthOutput:
    """Health check endpoint."""
    return HealthOutput(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        uptime_seconds=round(time.time() - start_time, 2),
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(transaction: TransactionInput) -> PredictionOutput:
    """Single transaction fraud prediction."""
    probabilities, is_fraud, threshold = predict_batch_vectorized([transaction])
    return PredictionOutput(
        fraud_probability=float(probabilities[0]),
        is_fraud=bool(is_fraud[0]),
        threshold_used=threshold,
    )


@app.post("/predict/batch", response_model=BatchPredictOutput)
async def predict_batch(batch: BatchPredictInput) -> BatchPredictOutput:
    """High-throughput vectorized batch fraud prediction."""
    probabilities, is_fraud, threshold = predict_batch_vectorized(batch.transactions)

    predictions = [
        PredictionOutput(
            fraud_probability=float(prob),
            is_fraud=bool(fraud),
            threshold_used=threshold,
        )
        for prob, fraud in zip(probabilities, is_fraud, strict=False)
    ]

    fraud_count = int(np.sum(is_fraud))
    summary = {
        "total_transactions": len(predictions),
        "fraud_detected": fraud_count,
        "fraud_rate": float(fraud_count / len(predictions)) if predictions else 0.0,
        "avg_fraud_probability": float(np.mean(probabilities)) if len(probabilities) > 0 else 0.0,
    }

    return BatchPredictOutput(predictions=predictions, summary=summary)


@app.get("/model/info", response_model=ModelInfoOutput)
async def model_info_endpoint() -> ModelInfoOutput:
    """Get loaded model metadata and deployment parameters."""
    if not model_info or model is None:
        raise HTTPException(status_code=503, detail="Model info not available")
    return ModelInfoOutput(**model_info)


if __name__ == "__main__":
    import uvicorn

    config = load_params()
    uvicorn.run(
        "mlops_fraud_detection.serve.main:app",
        host=config.serving.host,
        port=int(config.serving.port),
        workers=int(config.serving.workers),
        log_level=str(config.serving.log_level).lower(),
        reload=False,
    )
