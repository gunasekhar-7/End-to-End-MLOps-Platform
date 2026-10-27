"""Data and model drift detection for fraud detection."""

import json
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy import stats

from mlops_fraud_detection.common.utils import Timer, ensure_dir, get_logger, load_params, save_json

logger = get_logger(__name__)

warnings.filterwarnings("ignore")


class DriftDetector:
    """Drift detection for fraud detection model."""

    def __init__(self, config: DictConfig):
        self.config = config
        self.psi_threshold = config.monitoring.psi_threshold
        self.ks_threshold = config.monitoring.ks_threshold
        self.prediction_drift_threshold = config.monitoring.prediction_drift_threshold
        self.model_staleness_days = config.monitoring.model_staleness_days

        # Paths
        self.reference_data_path = config.data.external_dir
        self.processed_data_path = config.data.processed_dir
        self.model_path = config.model.model_file
        self.preprocessor_path = config.model.preprocessor_file
        self.metrics_path = config.metrics.train_file

        # State
        self.reference_data = None
        self.current_data = None
        self.model = None
        self.preprocessor = None
        self.baseline_metrics = None

    def load_reference_data(self) -> pd.DataFrame:
        """Load reference data for drift comparison."""
        # Try to load from external data directory
        ref_path = Path(self.reference_data_path) / "reference.parquet"
        if ref_path.exists():
            logger.info("Loading reference data", path=str(ref_path))
            return pd.read_parquet(ref_path)

        # Fallback: use training data as reference
        train_path = Path(self.config.data.train_file)
        if train_path.exists():
            logger.info("Using training data as reference", path=str(train_path))
            return pd.read_parquet(train_path)

        logger.warning("No reference data found")
        return None

    def load_current_data(self, days_back: int = 7) -> pd.DataFrame:
        """Load recent data for drift detection.

        In production, this would query a database or data lake.
        For now, we simulate with test data.
        """
        test_path = Path(self.config.data.test_file)
        if test_path.exists():
            logger.info("Using test data as current data", path=str(test_path))
            return pd.read_parquet(test_path)

        logger.warning("No current data found")
        return None

    def load_model_and_preprocessor(self):
        """Load model and preprocessor."""
        try:
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)
            logger.info("Model and preprocessor loaded")
        except Exception as e:
            logger.error("Failed to load model/preprocessor", error=str(e))
            raise

    def load_baseline_metrics(self) -> dict:
        """Load baseline metrics from training."""
        if Path(self.metrics_path).exists():
            with open(self.metrics_path) as f:
                return json.load(f)
        return {}

    def calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)."""
        # Handle edge cases
        if len(reference) == 0 or len(current) == 0:
            return 0.0

        # Create bins based on reference data quantiles
        try:
            _, bin_edges = np.histogram(reference, bins=bins)
            # Extend edges to cover all current values
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
        except Exception:
            return 0.0

        # Calculate distributions
        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        cur_hist, _ = np.histogram(current, bins=bin_edges)

        # Convert to percentages (add small epsilon to avoid division by zero)
        eps = 1e-6
        ref_perc = (ref_hist + eps) / (len(reference) + eps * bins)
        cur_perc = (cur_hist + eps) / (len(current) + eps * bins)

        # Calculate PSI
        psi = np.sum((cur_perc - ref_perc) * np.log(cur_perc / ref_perc))
        return float(psi)

    def calculate_ks_statistic(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate Kolmogorov-Smirnov statistic."""
        if len(reference) == 0 or len(current) == 0:
            return 0.0
        ks_stat, _ = stats.ks_2samp(reference, current)
        return float(ks_stat)

    def detect_data_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
        """Detect data drift using PSI and KS test on features."""
        logger.info("Detecting data drift")

        # Get numeric columns (excluding target)
        target_col = self.config.data.target_column
        numeric_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        drift_results = {
            "features_checked": len(numeric_cols),
            "drift_detected": False,
            "drifted_features": [],
            "feature_drift_scores": {},
        }

        for col in numeric_cols:
            ref_vals = reference_df[col].dropna().values
            cur_vals = current_df[col].dropna().values

            if len(ref_vals) == 0 or len(cur_vals) == 0:
                continue

            psi = self.calculate_psi(ref_vals, cur_vals)
            ks = self.calculate_ks_statistic(ref_vals, cur_vals)

            drift_results["feature_drift_scores"][col] = {
                "psi": psi,
                "ks": ks,
                "psi_alert": psi > self.psi_threshold,
                "ks_alert": ks > self.ks_threshold,
            }

            if psi > self.psi_threshold or ks > self.ks_threshold:
                drift_results["drift_detected"] = True
                drift_results["drifted_features"].append(
                    {
                        "feature": col,
                        "psi": psi,
                        "ks": ks,
                    }
                )

        logger.info(
            "Data drift detection complete",
            **{k: v for k, v in drift_results.items() if k != "feature_drift_scores"},
        )
        return drift_results

    def detect_prediction_drift(self, current_df: pd.DataFrame) -> dict:
        """Detect prediction drift by comparing prediction distributions."""
        logger.info("Detecting prediction drift")

        if self.model is None or self.preprocessor is None:
            self.load_model_and_preprocessor()

        target_col = self.config.data.target_column
        X_current = (
            current_df.drop(columns=[target_col])
            if target_col in current_df.columns
            else current_df
        )

        # Preprocess
        try:
            X_processed = self.preprocessor.transform(X_current)
        except Exception as e:
            logger.error("Preprocessing failed for prediction drift", error=str(e))
            return {"error": str(e)}

        # Get predictions
        preds = self.model.predict_proba(X_processed)[:, 1]

        # Compare with baseline prediction distribution
        baseline_metrics = self.load_baseline_metrics()
        baseline_fraud_rate = baseline_metrics.get("fraud_rate", 0.5)  # fallback

        current_fraud_rate = float(np.mean(preds))
        diff_std = abs(current_fraud_rate - baseline_fraud_rate) / (np.std(preds) + 1e-6)

        drift_results = {
            "baseline_fraud_rate": baseline_fraud_rate,
            "current_fraud_rate": current_fraud_rate,
            "std_deviations": diff_std,
            "drift_detected": diff_std > self.prediction_drift_threshold,
            "threshold": self.prediction_drift_threshold,
        }

        logger.info("Prediction drift detection complete", **drift_results)
        return drift_results

    def check_model_staleness(self) -> dict:
        """Check if model is stale (not retrained recently)."""
        logger.info("Checking model staleness")

        # Get model file modification time
        model_path = Path(self.model_path)
        if not model_path.exists():
            return {"stale": True, "reason": "Model file not found"}

        mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
        age_days = (datetime.now() - mtime).days

        result = {
            "model_age_days": age_days,
            "staleness_threshold_days": self.model_staleness_days,
            "stale": age_days > self.model_staleness_days,
            "last_trained": mtime.isoformat(),
        }

        logger.info("Model staleness check complete", **result)
        return result

    def run_drift_checks(self) -> dict:
        """Run all drift checks."""
        logger.info("Starting drift detection checks")

        with Timer("drift_detection", logger):
            results = {
                "timestamp": datetime.now().isoformat(),
                "data_drift": {},
                "prediction_drift": {},
                "model_staleness": {},
                "overall_drift_detected": False,
            }

            # Load data
            self.reference_data = self.load_reference_data()
            self.current_data = self.load_current_data()

            if self.reference_data is not None and self.current_data is not None:
                # Data drift
                results["data_drift"] = self.detect_data_drift(
                    self.reference_data, self.current_data
                )

                # Prediction drift
                results["prediction_drift"] = self.detect_prediction_drift(self.current_data)
            else:
                logger.warning("Skipping drift checks - insufficient data")
                results["data_drift"] = {"error": "Insufficient data"}
                results["prediction_drift"] = {"error": "Insufficient data"}

            # Model staleness
            results["model_staleness"] = self.check_model_staleness()

            # Overall drift flag
            results["overall_drift_detected"] = (
                results["data_drift"].get("drift_detected", False)
                or results["prediction_drift"].get("drift_detected", False)
                or results["model_staleness"].get("stale", False)
            )

            # Save results
            ensure_dir("reports/drift")
            save_json(results, "reports/drift/drift_report.json")

            logger.info("Drift detection complete", overall_drift=results["overall_drift_detected"])
            return results


def main():
    """Main drift detection entry point."""
    from mlops_fraud_detection.common.utils import setup_logging

    setup_logging("INFO")
    config = load_params()

    detector = DriftDetector(config)
    results = detector.run_drift_checks()

    # Print summary
    print("\n=== Drift Detection Report ===")
    print(f"Timestamp: {results['timestamp']}")
    print(f"Overall Drift Detected: {results['overall_drift_detected']}")
    print(f"\nData Drift: {results['data_drift'].get('drift_detected', 'N/A')}")
    if results["data_drift"].get("drifted_features"):
        print(f"  Drifted features: {len(results['data_drift']['drifted_features'])}")
        for f in results["data_drift"]["drifted_features"][:5]:
            print(f"    - {f['feature']}: PSI={f['psi']:.4f}, KS={f['ks']:.4f}")
    print(f"\nPrediction Drift: {results['prediction_drift'].get('drift_detected', 'N/A')}")
    if "std_deviations" in results["prediction_drift"]:
        print(
            f"  Std deviations from baseline: {results['prediction_drift']['std_deviations']:.2f}"
        )
    print(f"\nModel Staleness: {results['model_staleness'].get('stale', 'N/A')}")
    if "model_age_days" in results["model_staleness"]:
        print(f"  Model age: {results['model_staleness']['model_age_days']} days")

    # Exit with code 1 if drift detected (for CI/CD integration)
    if results["overall_drift_detected"]:
        print("\n⚠️  DRIFT DETECTED - Consider retraining")
        return 1
    else:
        print("\n✅ No significant drift detected")
        return 0


if __name__ == "__main__":
    exit(main())
