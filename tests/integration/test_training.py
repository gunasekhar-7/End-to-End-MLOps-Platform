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

        # Check metrics
        assert "f1" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics

        # Save model
        joblib.dump(model, config.model.model_file)
        assert Path(config.model.model_file).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
