"""Training script for fraud detection model.

This script:
1. Loads processed data from DVC outputs
2. Trains the model using parameters from params.yaml
3. Evaluates on validation set
4. Logs metrics and artifacts to MLflow
5. Saves model and metrics for DVC
"""

from pathlib import Path

import joblib
import mlflow.sklearn
import pandas as pd
from omegaconf import DictConfig, OmegaConf

import mlflow
from mlops_fraud_detection.common.utils import (
    Timer,
    calculate_classification_metrics,
    check_deployment_gates,
    get_git_sha,
    get_logger,
    load_params,
    save_json,
)

logger = get_logger(__name__)


def load_processed_data(train_path: str, val_path: str, target_column: str) -> tuple:
    """Load processed train and validation data."""
    logger.info("Loading processed data", train_path=train_path, val_path=val_path)

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_val = val_df.drop(columns=[target_column])
    y_val = val_df[target_column]

    logger.info(
        "Data loaded",
        train_shape=X_train.shape,
        val_shape=X_val.shape,
        train_target_dist=y_train.value_counts().to_dict(),
        val_target_dist=y_val.value_counts().to_dict(),
    )

    return X_train, y_train, X_val, y_val


def create_model(config: DictConfig):
    """Create model instance based on config."""
    model_type = config.training.model_type
    params = config.training

    logger.info("Creating model", model_type=model_type)

    if model_type == "xgboost":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            learning_rate=params.learning_rate,
            subsample=params.subsample,
            colsample_bytree=params.colsample_bytree,
            random_state=params.random_state,
            n_jobs=params.n_jobs,
            eval_metric=params.eval_metric,
            early_stopping_rounds=params.early_stopping_rounds,
            enable_categorical=False,
            tree_method="hist",
        )
    elif model_type == "lightgbm":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            learning_rate=params.learning_rate,
            subsample=params.subsample,
            colsample_bytree=params.colsample_bytree,
            random_state=params.random_state,
            n_jobs=params.n_jobs,
            verbose=-1,
        )
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            random_state=params.random_state,
            n_jobs=params.n_jobs,
        )
    elif model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            max_iter=1000,
            random_state=params.random_state,
            n_jobs=params.n_jobs,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def train_model(model, X_train, y_train, X_val, y_val, config: DictConfig):
    """Train model with early stopping if supported."""
    logger.info("Training model")

    model_type = config.training.model_type

    if model_type in ["xgboost", "lightgbm"]:
        # These support early stopping with eval_set
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train)

    logger.info("Model training complete")
    return model


def evaluate_model(model, X_val, y_val, threshold: float = 0.5) -> dict:
    """Evaluate model on validation set."""
    logger.info("Evaluating model")

    # Get predictions
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate metrics
    metrics = calculate_classification_metrics(y_val, y_pred, y_pred_proba)

    logger.info("Validation metrics", **metrics)
    return metrics, y_pred, y_pred_proba


def log_to_mlflow(
    model,
    metrics: dict,
    config: DictConfig,
    git_sha: str,
    preprocessor_path: str,
    X_val,
    y_val,
    y_pred,
    y_pred_proba,
):
    """Log model, metrics, and artifacts to MLflow."""
    logger.info("Logging to MLflow")

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Log parameters
        mlflow.log_params(OmegaConf.to_container(config.training, resolve=True))
        mlflow.log_params(OmegaConf.to_container(config.preprocessing, resolve=True))
        mlflow.log_param("git_sha", git_sha)
        mlflow.log_param("data_version", get_git_sha())  # Use git SHA as data version proxy

        # Log metrics
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(name, value)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=config.mlflow.model_name,
        )

        # Log preprocessor
        mlflow.log_artifact(preprocessor_path, "preprocessor")

        # Log confusion matrix plot
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay

            fig, ax = plt.subplots(figsize=(6, 6))
            ConfusionMatrixDisplay.from_predictions(y_val, y_pred, ax=ax)
            ax.set_title("Confusion Matrix (Validation)")
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)
        except Exception as e:
            logger.warning("Failed to log confusion matrix", error=str(e))

        # Log ROC curve
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import RocCurveDisplay

            fig, ax = plt.subplots(figsize=(6, 6))
            RocCurveDisplay.from_predictions(y_val, y_pred_proba, ax=ax)
            ax.set_title("ROC Curve (Validation)")
            mlflow.log_figure(fig, "roc_curve.png")
            plt.close(fig)
        except Exception as e:
            logger.warning("Failed to log ROC curve", error=str(e))

        # Log PR curve
        try:
            import matplotlib.pyplot as plt
            from sklearn.metrics import PrecisionRecallDisplay

            fig, ax = plt.subplots(figsize=(6, 6))
            PrecisionRecallDisplay.from_predictions(y_val, y_pred_proba, ax=ax)
            ax.set_title("Precision-Recall Curve (Validation)")
            mlflow.log_figure(fig, "pr_curve.png")
            plt.close(fig)
        except Exception as e:
            logger.warning("Failed to log PR curve", error=str(e))

        logger.info("MLflow logging complete", run_id=run_id)
        return run_id


def save_model_and_metrics(model, metrics: dict, config: DictConfig):
    """Save model to models/ and metrics to metrics/ for DVC."""
    logger.info("Saving model and metrics")

    # Save model
    model_path = config.model.model_file
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    # Save metrics
    metrics_path = config.metrics.train_file
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)

    # Add metadata to metrics
    metrics_with_meta = {
        **metrics,
        "git_sha": get_git_sha(),
        "model_type": config.training.model_type,
        "threshold": config.serving.threshold,
    }
    save_json(metrics_with_meta, metrics_path)

    # Also save validation metrics separately
    val_metrics_path = config.metrics.val_file
    save_json(metrics_with_meta, val_metrics_path)

    logger.info("Model and metrics saved", model_path=model_path, metrics_path=metrics_path)


def main():
    """Main training pipeline."""
    from mlops_fraud_detection.common.utils import setup_logging

    setup_logging("INFO")

    # Load configuration
    config = load_params()

    with Timer("training_pipeline", logger):
        # Load data
        X_train, y_train, X_val, y_val = load_processed_data(
            config.data.train_file, config.data.val_file, config.data.target_column
        )

        # Load preprocessor (for reference)
        preprocessor = joblib.load(config.model.preprocessor_file)

        # Create and train model
        model = create_model(config)
        model = train_model(model, X_train, y_train, X_val, y_val, config)

        # Evaluate
        metrics, y_pred, y_pred_proba = evaluate_model(
            model, X_val, y_val, config.serving.threshold
        )

        # Check deployment gates
        gates_passed, gate_checks = check_deployment_gates(metrics, config)
        metrics["deployment_gates_passed"] = gates_passed
        metrics["gate_checks"] = gate_checks

        logger.info("Deployment gate check", passed=gates_passed, checks=gate_checks)

        if not gates_passed:
            logger.warning("Model does not meet deployment thresholds!")
            for metric, passed in gate_checks.items():
                status = "PASS" if passed else "FAIL"
                logger.info(f"  {metric}: {status}")

        # Save model and metrics
        save_model_and_metrics(model, metrics, config)

        # Log to MLflow
        try:
            log_to_mlflow(
                model,
                metrics,
                config,
                get_git_sha(),
                config.model.preprocessor_file,
                X_val,
                y_val,
                y_pred,
                y_pred_proba,
            )
        except Exception as e:
            logger.warning("MLflow logging failed (server may not be running)", error=str(e))

        logger.info("Training pipeline complete", metrics=metrics)

    return metrics


if __name__ == "__main__":
    main()
