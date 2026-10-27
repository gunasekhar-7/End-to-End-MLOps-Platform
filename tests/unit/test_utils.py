"""Unit tests for common utilities."""

import pytest
from omegaconf import OmegaConf

from mlops_fraud_detection.common.utils import (
    Timer,
    calculate_classification_metrics,
    check_deployment_gates,
    load_json,
    save_json,
)


class TestClassificationMetrics:
    """Tests for calculate_classification_metrics."""

    def test_perfect_predictions(self):
        y_true = [0, 1, 1, 0, 1]
        y_pred = [0, 1, 1, 0, 1]
        y_proba = [0.1, 0.9, 0.8, 0.2, 0.7]

        metrics = calculate_classification_metrics(y_true, y_pred, y_proba)

        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["roc_auc"] == 1.0
        assert metrics["pr_auc"] == 1.0

    def test_all_wrong_predictions(self):
        y_true = [0, 1, 1, 0, 1]
        y_pred = [1, 0, 0, 1, 0]
        y_proba = [0.9, 0.1, 0.2, 0.8, 0.3]

        metrics = calculate_classification_metrics(y_true, y_pred, y_proba)

        assert metrics["accuracy"] == 0.0
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0

    def test_mixed_predictions(self):
        y_true = [0, 0, 1, 1, 0, 1, 0, 1]
        y_pred = [0, 0, 1, 0, 0, 1, 1, 1]
        y_proba = [0.1, 0.2, 0.8, 0.4, 0.3, 0.9, 0.6, 0.7]

        metrics = calculate_classification_metrics(y_true, y_pred, y_proba)

        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1
        assert 0 <= metrics["pr_auc"] <= 1
        assert "true_positives" in metrics
        assert "false_positives" in metrics

    def test_without_probabilities(self):
        y_true = [0, 1, 1, 0]
        y_pred = [0, 1, 0, 0]

        metrics = calculate_classification_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "roc_auc" not in metrics
        assert "pr_auc" not in metrics


class TestDeploymentGates:
    """Tests for check_deployment_gates."""

    def test_all_gates_pass(self):
        metrics = {
            "f1": 0.90,
            "precision": 0.85,
            "recall": 0.85,
            "pr_auc": 0.90,
            "roc_auc": 0.95,
        }
        params = OmegaConf.create(
            {
                "deployment_gates": {
                    "min_f1": 0.85,
                    "min_precision": 0.80,
                    "min_recall": 0.80,
                    "min_pr_auc": 0.85,
                    "min_roc_auc": 0.90,
                }
            }
        )

        passed, checks = check_deployment_gates(metrics, params)

        assert passed is True
        assert all(checks.values())

    def test_some_gates_fail(self):
        metrics = {
            "f1": 0.80,  # Below 0.85
            "precision": 0.85,
            "recall": 0.85,
            "pr_auc": 0.90,
            "roc_auc": 0.95,
        }
        params = OmegaConf.create(
            {
                "deployment_gates": {
                    "min_f1": 0.85,
                    "min_precision": 0.80,
                    "min_recall": 0.80,
                    "min_pr_auc": 0.85,
                    "min_roc_auc": 0.90,
                }
            }
        )

        passed, checks = check_deployment_gates(metrics, params)

        assert passed is False
        assert checks["f1"] is False
        assert checks["precision"] is True

    def test_edge_case_exact_threshold(self):
        metrics = {
            "f1": 0.85,
            "precision": 0.80,
            "recall": 0.80,
            "pr_auc": 0.85,
            "roc_auc": 0.90,
        }
        params = OmegaConf.create(
            {
                "deployment_gates": {
                    "min_f1": 0.85,
                    "min_precision": 0.80,
                    "min_recall": 0.80,
                    "min_pr_auc": 0.85,
                    "min_roc_auc": 0.90,
                }
            }
        )

        passed, checks = check_deployment_gates(metrics, params)

        assert passed is True
        assert all(checks.values())


class TestJSONUtils:
    """Tests for JSON save/load."""

    def test_save_load_json(self, tmp_path):
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        file_path = tmp_path / "test.json"

        save_json(data, str(file_path))
        loaded = load_json(str(file_path))

        assert loaded == data

    def test_save_creates_directory(self, tmp_path):
        data = {"test": "data"}
        file_path = tmp_path / "subdir" / "test.json"

        save_json(data, str(file_path))

        assert file_path.exists()
        assert load_json(str(file_path)) == data


class TestTimer:
    """Tests for Timer context manager."""

    def test_timer_measures_time(self):
        with Timer("test_operation") as timer:
            import time

            time.sleep(0.01)

        assert timer.elapsed is not None
        assert timer.elapsed >= 0.01
        assert timer.elapsed < 1.0  # Should be much less than 1 second


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
