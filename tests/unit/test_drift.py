"""Unit tests for drift detection module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlops_fraud_detection.monitor.drift import DriftDetector


@pytest.fixture
def mock_config(tmp_path):
    from omegaconf import OmegaConf

    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    external_dir = data_dir / "external"
    models_dir = tmp_path / "models"
    metrics_dir = tmp_path / "metrics"

    for d in [raw_dir, processed_dir, external_dir, models_dir, metrics_dir]:
        d.mkdir(parents=True, exist_ok=True)

    return OmegaConf.create(
        {
            "data": {
                "raw_path": str(raw_dir / "fraud_data.csv"),
                "processed_dir": str(processed_dir),
                "train_file": str(processed_dir / "train.parquet"),
                "val_file": str(processed_dir / "val.parquet"),
                "test_file": str(processed_dir / "test.parquet"),
                "external_dir": str(external_dir),
                "target_column": "is_fraud",
            },
            "model": {
                "output_dir": str(models_dir),
                "model_file": str(models_dir / "model.pkl"),
                "preprocessor_file": str(models_dir / "preprocessor.pkl"),
            },
            "metrics": {
                "train_file": str(metrics_dir / "train.json"),
                "val_file": str(metrics_dir / "val.json"),
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


class TestDriftStatistics:
    """Test mathematical statistics functions."""

    def test_psi_identical_distribution(self, mock_config):
        detector = DriftDetector(mock_config)
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(0, 1, 1000)

        psi = detector.calculate_psi(ref, cur, bins=10)
        assert psi < 0.1  # Virtually no drift

    def test_psi_shifted_distribution(self, mock_config):
        detector = DriftDetector(mock_config)
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(3, 1, 1000)  # Heavy shift

        psi = detector.calculate_psi(ref, cur, bins=10)
        assert psi > mock_config.monitoring.psi_threshold

    def test_psi_empty_arrays(self, mock_config):
        detector = DriftDetector(mock_config)
        psi = detector.calculate_psi(np.array([]), np.array([]))
        assert psi == 0.0

    def test_ks_statistic_identical(self, mock_config):
        detector = DriftDetector(mock_config)
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(0, 1, 1000)

        ks = detector.calculate_ks_statistic(ref, cur)
        assert ks < 0.1

    def test_ks_statistic_different(self, mock_config):
        detector = DriftDetector(mock_config)
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        cur = np.random.normal(2, 1, 1000)

        ks = detector.calculate_ks_statistic(ref, cur)
        assert ks > mock_config.monitoring.ks_threshold

    def test_ks_empty_arrays(self, mock_config):
        detector = DriftDetector(mock_config)
        ks = detector.calculate_ks_statistic(np.array([]), np.array([]))
        assert ks == 0.0


class TestDriftDetectorWorkflow:
    """Test full drift detector workflow."""

    def test_detect_data_drift_no_drift(self, mock_config):
        detector = DriftDetector(mock_config)
        np.random.seed(42)
        ref_df = pd.DataFrame(
            {
                "feat1": np.random.normal(10, 2, 500),
                "feat2": np.random.uniform(0, 100, 500),
                "is_fraud": np.random.choice([0, 1], 500),
            }
        )
        cur_df = ref_df.copy()

        results = detector.detect_data_drift(ref_df, cur_df)
        assert results["drift_detected"] is False
        assert len(results["drifted_features"]) == 0
        assert "feat1" in results["feature_drift_scores"]

    def test_detect_data_drift_with_drift(self, mock_config):
        detector = DriftDetector(mock_config)
        np.random.seed(42)
        ref_df = pd.DataFrame(
            {
                "feat1": np.random.normal(10, 2, 500),
                "is_fraud": np.random.choice([0, 1], 500),
            }
        )
        cur_df = pd.DataFrame(
            {
                "feat1": np.random.normal(25, 2, 500),  # Dramatic drift
                "is_fraud": np.random.choice([0, 1], 500),
            }
        )

        results = detector.detect_data_drift(ref_df, cur_df)
        assert results["drift_detected"] is True
        assert len(results["drifted_features"]) == 1
        assert results["drifted_features"][0]["feature"] == "feat1"

    def test_model_staleness_check(self, mock_config):
        detector = DriftDetector(mock_config)

        # Model not found
        res_missing = detector.check_model_staleness()
        assert res_missing["stale"] is True

        # Existing fresh model
        model_file = Path(mock_config.model.model_file)
        model_file.write_text("dummy model")
        res_fresh = detector.check_model_staleness()
        assert res_fresh["stale"] is False
        assert res_fresh["model_age_days"] == 0

    def test_prediction_drift_and_full_run(self, mock_config):
        detector = DriftDetector(mock_config)

        class MockModel:
            def predict_proba(self, X):
                return np.array([[0.8, 0.2]] * len(X))

        class MockPreproc:
            def transform(self, df):
                return df.fillna(0.0).values

        detector.model = MockModel()
        detector.preprocessor = MockPreproc()

        df = pd.DataFrame({"feat1": [1.0, 2.0, 3.0], "is_fraud": [0, 1, 0]})
        pred_drift = detector.detect_prediction_drift(df)
        assert "current_fraud_rate" in pred_drift
        assert np.isclose(pred_drift["current_fraud_rate"], 0.2)

        # Test run_drift_checks
        train_file = Path(mock_config.data.train_file)
        test_file = Path(mock_config.data.test_file)
        df.to_parquet(train_file)
        df.to_parquet(test_file)

        full_results = detector.run_drift_checks()
        assert "timestamp" in full_results
        assert "overall_drift_detected" in full_results

    def test_data_loaders(self, mock_config):
        detector = DriftDetector(mock_config)
        df = pd.DataFrame({"feat1": [10.0], "is_fraud": [0]})

        # Test when files don't exist
        assert detector.load_reference_data() is None
        assert detector.load_current_data() is None

        # Test when files exist
        ref_file = Path(mock_config.data.external_dir) / "reference.parquet"
        test_file = Path(mock_config.data.test_file)
        df.to_parquet(ref_file)
        df.to_parquet(test_file)

        assert detector.load_reference_data() is not None
        assert detector.load_current_data() is not None


def test_drift_main_entrypoint(mock_config, monkeypatch):
    from mlops_fraud_detection.monitor import drift as drift_module

    monkeypatch.setattr(drift_module, "load_params", lambda: mock_config)

    class DummyDetector:
        def __init__(self, cfg):
            pass

        def run_drift_checks(self):
            return {
                "timestamp": "2026-10-31T00:00:00",
                "overall_drift_detected": False,
                "data_drift": {"drift_detected": False, "drifted_features": []},
                "prediction_drift": {"drift_detected": False, "std_deviations": 0.5},
                "model_staleness": {"stale": False, "model_age_days": 2},
            }

    monkeypatch.setattr(drift_module, "DriftDetector", DummyDetector)
    code = drift_module.main()
    assert code == 0
