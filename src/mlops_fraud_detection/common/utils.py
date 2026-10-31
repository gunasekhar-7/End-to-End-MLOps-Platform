"""Shared utilities for the MLOps Fraud Detection platform."""

import json
import logging
from pathlib import Path
from typing import Any, cast

import structlog
from omegaconf import DictConfig, OmegaConf

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)


def get_logger(name: str) -> Any:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def load_params(config_path: str = "params.yaml") -> DictConfig:
    """Load parameters from params.yaml using OmegaConf."""
    config = OmegaConf.load(config_path)
    if not isinstance(config, DictConfig):
        raise TypeError(f"Expected DictConfig from {config_path}, got {type(config).__name__}")
    return cast(DictConfig, config)


def save_json(data: dict[str, Any], path: str) -> None:
    """Save dictionary to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> dict[str, Any]:
    """Load JSON file to dictionary."""
    with Path(path).open() as f:
        data: dict[str, Any] = json.load(f)
        return data


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_dvc_data_hash(data_path: str = "data/raw/fraud_data.csv") -> str:
    """Get DVC hash for data file."""
    try:
        import subprocess

        result = subprocess.run(
            ["dvc", "status", "-c", data_path], capture_output=True, text=True, check=True
        )
        # Parse output to get hash
        return result.stdout.strip().split()[-1] if result.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def setup_logging(log_level: str = "INFO") -> None:
    """Configure standard library logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(message)s",
    )


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str, logger: Any = None):
        self.name = name
        self.logger = logger or get_logger(__name__)
        self.start_time: float | None = None
        self.elapsed: float | None = None

    def __enter__(self) -> "Timer":
        import time

        self.start_time = time.perf_counter()
        self.logger.info(f"{self.name} started")
        return self

    def __exit__(self, *args: Any) -> None:
        import time

        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time
            self.logger.info(f"{self.name} completed", elapsed_seconds=round(self.elapsed, 3))


def calculate_classification_metrics(
    y_true: Any, y_pred: Any, y_proba: Any = None
) -> dict[str, float]:
    """Calculate standard classification metrics."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        try:
            import numpy as np

            auc = float(roc_auc_score(y_true, y_proba))
            metrics["roc_auc"] = 0.0 if np.isnan(auc) else auc
            pr = float(average_precision_score(y_true, y_proba))
            metrics["pr_auc"] = 0.0 if np.isnan(pr) else pr
        except (ValueError, Exception):
            # When only one class is present in true labels
            metrics["roc_auc"] = 0.0
            metrics["pr_auc"] = 0.0

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update(
            {
                "true_negatives": float(tn),
                "false_positives": float(fp),
                "false_negatives": float(fn),
                "true_positives": float(tp),
            }
        )

    return metrics


def check_deployment_gates(
    metrics: dict[str, float], params: DictConfig
) -> tuple[bool, dict[str, bool]]:
    """Check if metrics meet deployment gate thresholds."""
    gates = params.deployment_gates
    checks = {
        "f1": metrics.get("f1", 0.0) >= gates.min_f1,
        "precision": metrics.get("precision", 0.0) >= gates.min_precision,
        "recall": metrics.get("recall", 0.0) >= gates.min_recall,
        "pr_auc": metrics.get("pr_auc", 0.0) >= gates.min_pr_auc,
        "roc_auc": metrics.get("roc_auc", 0.0) >= gates.min_roc_auc,
    }

    all_passed = bool(all(checks.values()))
    return all_passed, checks
