"""Integration tests for training pipeline."""

from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.integration
class TestTrainingPipeline:
    """Integration tests for the training pipeline."""

    def test_prepare_data_creates_outputs(self, sample_data, sample_config, tmp_path):
        """Test that prepare_data creates expected output files."""
        from mlops_fraud_detection.train.prepare_data import prepare_data

        # Setup temp directory structure
        data_dir = tmp_path / "data"
        raw_dir = data_dir / "raw"
        processed_dir = data_dir / "processed"
        models_dir = tmp_path / "models"
        reports_dir = tmp_path / "reports" / "data_validation"

        raw_dir.mkdir(parents=True)
        processed_dir.mkdir(parents=True)
        models_dir.mkdir(parents=True)
        reports_dir.mkdir(parents=True)

        # Save sample data
        raw_file = raw_dir / "fraud_data.csv"
        sample_data.to_csv(raw_file, index=False)

        # Update config paths
        config = sample_config
        config.data.raw_path = str(raw_file)
        config.data.processed_dir = str(processed_dir)
        config.data.train_file = str(processed_dir / "train.parquet")
        config.data.val_file = str(processed_dir / "val.parquet")
        config.data.test_file = str(processed_dir / "test.parquet")
        config.model.preprocessor_file = str(models_dir / "preprocessor.pkl")

        # Run prepare_data
        result = prepare_data(config)

        # Check outputs
        assert Path(config.data.train_file).exists()
        assert Path(config.data.val_file).exists()
        assert Path(config.data.test_file).exists()
        assert Path(config.model.preprocessor_file).exists()
        assert Path("reports/data_validation/validation_report.json").exists()

        # Check result
        assert "train_shape" in result
        assert "val_shape" in result
        assert "test_shape" in result
        assert "n_features" in result
        assert result["n_features"] > 0

    def test_train_creates_model(self, sample_data, sample_config, tmp_path):
        """Test that train creates model and metrics."""
        import joblib

        from mlops_fraud_detection.train.prepare_data import prepare_data
        from mlops_fraud_detection.train.train import create_model, evaluate_model, train_model

        # Setup
        data_dir = tmp_path / "data"
        processed_dir = data_dir / "processed"
        models_dir = tmp_path / "models"
        metrics_dir = tmp_path / "metrics"

        processed_dir.mkdir(parents=True)
        models_dir.mkdir(parents=True)
        metrics_dir.mkdir(parents=True)

        # Prepare data first
        config = sample_config
        config.data.processed_dir = str(processed_dir)
        config.data.train_file = str(processed_dir / "train.parquet")
        config.data.val_file = str(processed_dir / "val.parquet")
        config.data.test_file = str(processed_dir / "test.parquet")
        config.model.preprocessor_file = str(models_dir / "preprocessor.pkl")
        config.model.model_file = str(models_dir / "model.pkl")
        config.metrics.train_file = str(metrics_dir / "train.json")

        # Save sample data
        raw_dir = data_dir / "raw"
        raw_dir.mkdir(parents=True)
        raw_file = raw_dir / "fraud_data.csv"
        sample_data.to_csv(raw_file, index=False)
        config.data.raw_path = str(raw_file)

        # Prepare
        prepare_data(config)

        # Load prepared data
        train_df = pd.read_parquet(config.data.train_file)
        val_df = pd.read_parquet(config.data.val_file)

        X_train = train_df.drop(columns=[config.data.target_column])
        y_train = train_df[config.data.target_column]
        X_val = val_df.drop(columns=[config.data.target_column])
        y_val = val_df[config.data.target_column]

        # Create and train model
        model = create_model(config)
        model = train_model(model, X_train, y_train, X_val, y_val, config)

        # Evaluate
        metrics, y_pred, y_proba = evaluate_model(model, X_val, y_val, config.serving.threshold)

        # Check metrics and predictions
        assert len(y_pred) == len(y_val)
        assert len(y_proba) == len(y_val)
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics

        # Save model
        joblib.dump(model, config.model.model_file)
        assert Path(config.model.model_file).exists()

        # Test save_model_and_metrics
        from mlops_fraud_detection.train.train import save_model_and_metrics

        save_model_and_metrics(model, metrics, config)
        assert Path(config.metrics.train_file).exists()
        assert Path(config.metrics.val_file).exists()

    def test_log_to_mlflow_mocked(self, sample_config, monkeypatch, tmp_path):
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        from mlops_fraud_detection.train.train import log_to_mlflow

        model = LogisticRegression()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])
        model.fit(X, y)

        preproc_path = tmp_path / "preprocessor.pkl"
        preproc_path.write_text("mock")

        # Mock mlflow calls
        monkeypatch.setattr("mlflow.set_tracking_uri", lambda uri: None)
        monkeypatch.setattr("mlflow.set_experiment", lambda name: None)

        class MockRunInfo:
            run_id = "test_run_123"

        class MockRun:
            info = MockRunInfo()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        monkeypatch.setattr("mlflow.start_run", lambda: MockRun())
        monkeypatch.setattr("mlflow.log_params", lambda params: None)
        monkeypatch.setattr("mlflow.log_param", lambda k, v: None)
        monkeypatch.setattr("mlflow.log_metric", lambda k, v: None)
        monkeypatch.setattr("mlflow.log_artifact", lambda path, artifact_path=None: None)
        monkeypatch.setattr("mlflow.log_figure", lambda *args, **kwargs: None)
        monkeypatch.setattr("mlflow.sklearn.log_model", lambda **kwargs: None)

        import pandas as pd

        df_x = pd.DataFrame(X, columns=["f1", "f2"])
        s_y = pd.Series(y)

        run_id = log_to_mlflow(
            model=model,
            metrics={"f1": 0.95},
            config=sample_config,
            git_sha="test_sha",
            preprocessor_path=str(preproc_path),
            X_val=df_x,
            y_val=s_y,
            y_pred=y,
            y_pred_proba=np.array([0.1, 0.9]),
        )
        assert run_id == "test_run_123"

    def test_model_types_and_load_data(self, sample_config, tmp_path):
        import copy

        from mlops_fraud_detection.train.train import create_model, load_processed_data, train_model

        # Test model creation for all supported types
        for m_type in ["random_forest", "logistic_regression", "lightgbm"]:
            cfg = copy.deepcopy(sample_config)
            cfg.training.model_type = m_type
            m = create_model(cfg)
            assert m is not None

        # Test train_model with random forest
        import pandas as pd

        X = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [5.0, 6.0, 7.0, 8.0]})
        y = pd.Series([0, 1, 0, 1])

        rf_cfg = copy.deepcopy(sample_config)
        rf_cfg.training.model_type = "random_forest"
        rf_model = create_model(rf_cfg)
        trained_rf = train_model(rf_model, X, y, X, y, rf_cfg)
        assert trained_rf is not None

        # Test load_processed_data
        train_path = tmp_path / "train.parquet"
        val_path = tmp_path / "val.parquet"
        data_df = pd.DataFrame({"feat": [1.0, 2.0], "is_fraud": [0, 1]})
        data_df.to_parquet(train_path)
        data_df.to_parquet(val_path)

        x_tr, y_tr, x_v, y_v = load_processed_data(str(train_path), str(val_path), "is_fraud")
        assert len(x_tr) == 2
        assert len(y_tr) == 2

    def test_train_main_execution(self, sample_config, monkeypatch, tmp_path):
        import copy

        import pandas as pd

        from mlops_fraud_detection.train import train as train_module

        train_path = tmp_path / "train.parquet"
        val_path = tmp_path / "val.parquet"
        model_path = tmp_path / "model.pkl"
        metrics_train = tmp_path / "train.json"
        metrics_val = tmp_path / "val.json"

        # Generate toy data
        df = pd.DataFrame(
            {
                "f1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "f2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "is_fraud": [0, 1, 0, 1, 0, 1],
            }
        )
        df.to_parquet(train_path)
        df.to_parquet(val_path)

        cfg = copy.deepcopy(sample_config)
        cfg.data.train_file = str(train_path)
        cfg.data.val_file = str(val_path)
        cfg.model.model_file = str(model_path)
        cfg.metrics.train_file = str(metrics_train)
        cfg.metrics.val_file = str(metrics_val)
        cfg.training.n_estimators = 2
        cfg.training.max_depth = 2

        monkeypatch.setattr(train_module, "load_params", lambda: cfg)
        monkeypatch.setattr(train_module, "log_to_mlflow", lambda *args, **kwargs: "mock_run")

        res_metrics = train_module.main()
        assert "f1" in res_metrics
        assert Path(model_path).exists()
        assert Path(metrics_train).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
