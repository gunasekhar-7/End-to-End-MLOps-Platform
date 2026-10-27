"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture(scope="session")
def sample_data():
    """Sample data for testing."""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    n = 1000

    return pd.DataFrame(
        {
            "amount": np.random.exponential(100, n),
            "time_since_last_tx": np.random.exponential(24, n),
            "merchant_category": np.random.choice(
                ["grocery", "online", "travel", "entertainment"], n
            ),
            "tx_type": np.random.choice(["online", "pos", "atm"], n),
            "country": np.random.choice(["US", "CA", "UK", "DE"], n),
            "is_fraud": np.random.choice([0, 1], n, p=[0.98, 0.02]),
        }
    )


@pytest.fixture(scope="session")
def sample_config():
    """Sample configuration for testing."""
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "data": {
                "raw_path": "data/raw/fraud_data.csv",
                "processed_dir": "data/processed",
                "train_file": "data/processed/train.parquet",
                "val_file": "data/processed/val.parquet",
                "test_file": "data/processed/test.parquet",
                "target_column": "is_fraud",
            },
            "model": {
                "output_dir": "models",
                "model_file": "models/model.pkl",
                "preprocessor_file": "models/preprocessor.pkl",
            },
            "metrics": {
                "train_file": "metrics/train.json",
                "val_file": "metrics/val.json",
            },
            "training": {
                "model_type": "xgboost",
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
                "n_jobs": 1,
                "early_stopping_rounds": 5,
                "eval_metric": "auc",
            },
            "preprocessing": {
                "test_size": 0.2,
                "val_size": 0.1,
                "stratify": True,
                "handle_missing": "median",
                "encode_categorical": "target",
                "scale_numeric": True,
            },
            "mlflow": {
                "tracking_uri": "http://localhost:5000",
                "experiment_name": "fraud-detection-test",
                "model_name": "fraud-detection-test",
            },
            "deployment_gates": {
                "min_f1": 0.5,
                "min_precision": 0.5,
                "min_recall": 0.5,
                "min_pr_auc": 0.5,
                "min_roc_auc": 0.5,
            },
            "serving": {
                "host": "0.0.0.0",
                "port": 8000,
                "threshold": 0.5,
                "log_level": "INFO",
                "workers": 1,
            },
            "monitoring": {
                "drift_check_interval_hours": 24,
                "psi_threshold": 0.2,
                "ks_threshold": 0.05,
                "prediction_drift_threshold": 2.0,
                "model_staleness_days": 30,
            },
        }
    )
