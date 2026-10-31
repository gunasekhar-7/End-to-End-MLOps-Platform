"""Data and model drift detection for fraud detection."""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

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
        self.psi_threshold = float(config.monitoring.psi_threshold)
        self.ks_threshold = float(config.monitoring.ks_threshold)
        self.prediction_drift_threshold = float(config.monitoring.prediction_drift_threshold)
        self.model_staleness_days = int(config.monitoring.model_staleness_days)

        # Paths
        self.reference_data_path = str(config.data.external_dir)
        self.processed_data_path = str(config.data.processed_dir)
        self.model_path = str(config.model.model_file)
        self.preprocessor_path = str(config.model.preprocessor_file)
        self.metrics_path = str(config.metrics.train_file)

        # State
        self.reference_data: pd.DataFrame | None = None
        self.current_data: pd.DataFrame | None = None
        self.model: Any = None
        self.preprocessor: Any = None
        self.baseline_metrics: dict[str, Any] | None = None

    def load_reference_data(self) -> pd.DataFrame | None:
        """Load reference data for drift comparison."""
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

    def load_current_data(self) -> pd.DataFrame | None:
        """Load recent data for drift detection."""
        test_path = Path(self.config.data.test_file)
        if test_path.exists():
            logger.info("Using test data as current data", path=str(test_path))
            return pd.read_parquet(test_path)

        logger.warning("No current data found")
        return None

    def load_model_and_preprocessor(self) -> None:
        """Load model and preprocessor."""
        try:
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)
            logger.info("Model and preprocessor loaded")
        except Exception as e:
            logger.error("Failed to load model/preprocessor", error=str(e))
            raise

    def load_baseline_metrics(self) -> dict[str, Any]:
        """Load baseline metrics from training."""
        if Path(self.metrics_path).exists():
            with Path(self.metrics_path).open() as f:
                data: dict[str, Any] = json.load(f)
                return data
        return {}

    def calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index (PSI)."""
        if len(reference) == 0 or len(current) == 0:
            return 0.0

        try:
            _, bin_edges = np.histogram(reference, bins=bins)
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf
        except Exception:
            return 0.0

        ref_hist, _ = np.histogram(reference, bins=bin_edges)
        cur_hist, _ = np.histogram(current, bins=bin_edges)

        eps = 1e-6
        ref_perc = (ref_hist + eps) / (len(reference) + eps * bins)
        cur_perc = (cur_hist + eps) / (len(current) + eps * bins)

        psi = np.sum((cur_perc - ref_perc) * np.log(cur_perc / ref_perc))
        return float(psi)

    def calculate_ks_statistic(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Calculate Kolmogorov-Smirnov statistic."""
        if len(reference) == 0 or len(current) == 0:
            return 0.0
        ks_stat, _ = stats.ks_2samp(reference, current)
        return float(ks_stat)

    def detect_data_drift(
        self, reference_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> dict[str, Any]:
        """Detect data drift using PSI and KS test on features."""
        logger.info("Detecting data drift")

        target_col = str(self.config.data.target_column)
        numeric_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)

        feature_scores: dict[str, Any] = {}
        drifted_features: list[dict[str, Any]] = []

        for col in numeric_cols:
            if col not in current_df.columns:
                continue

            ref_vals = reference_df[col].dropna().values
            cur_vals = current_df[col].dropna().values

            if len(ref_vals) == 0 or len(cur_vals) == 0:
                continue

            psi = self.calculate_psi(ref_vals, cur_vals)
            ks = self.calculate_ks_statistic(ref_vals, cur_vals)

            feature_scores[col] = {
                "psi": psi,
                "ks": ks,
                "psi_alert": psi > self.psi_threshold,
                "ks_alert": ks > self.ks_threshold,
            }

            if psi > self.psi_threshold or ks > self.ks_threshold:
                drifted_features.append(
                    {
                        "feature": col,
                        "psi": psi,
                        "ks": ks,
                    }
                )

        drift_results: dict[str, Any] = {
            "features_checked": len(feature_scores),
            "drift_detected": len(drifted_features) > 0,
            "drifted_features": drifted_features,
            "feature_drift_scores": feature_scores,
        }

        logger.info(
            "Data drift detection complete",
            drift_detected=drift_results["drift_detected"],
            drifted_count=len(drifted_features),
        )
        return drift_results

    def detect_prediction_drift(self, current_df: pd.DataFrame) -> dict[str, Any]:
        """Detect prediction drift by comparing prediction distributions."""
        logger.info("Detecting prediction drift")

        if self.model is None or self.preprocessor is None:
            self.load_model_and_preprocessor()

        target_col = str(self.config.data.target_column)
        X_current = (
            current_df.drop(columns=[target_col])
            if target_col in current_df.columns
            else current_df
        )

        try:
            X_processed = self.preprocessor.transform(X_current)
        except Exception as e:
            logger.error("Preprocessing failed for prediction drift", error=str(e))
            return {"error": str(e), "drift_detected": False}

        preds = self.model.predict_proba(X_processed)[:, 1]
        baseline_metrics = self.load_baseline_metrics()

        if not baseline_metrics or "fraud_rate" not in baseline_metrics:
            baseline_fraud_rate = float(np.mean(preds))
            diff_std = 0.0
            drift_detected = False
        else:
            baseline_fraud_rate = float(baseline_metrics.get("fraud_rate", 0.5))
            current_fraud_rate = float(np.mean(preds))
            diff_std = float(abs(current_fraud_rate - baseline_fraud_rate) / (np.std(preds) + 1e-6))
            drift_detected = diff_std > self.prediction_drift_threshold

        drift_results: dict[str, Any] = {
            "baseline_fraud_rate": baseline_fraud_rate,
            "current_fraud_rate": float(np.mean(preds)),
            "std_deviations": diff_std,
            "drift_detected": drift_detected,
            "threshold": self.prediction_drift_threshold,
        }

        logger.info("Prediction drift detection complete", **drift_results)
        return drift_results

    def check_model_staleness(self) -> dict[str, Any]:
        """Check if model is stale (not retrained recently)."""
        logger.info("Checking model staleness")

        model_path = Path(self.model_path)
        if not model_path.exists():
            return {"stale": True, "reason": "Model file not found"}

        mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
        age_days = max(0, int((datetime.now() - mtime).total_seconds() / 86400))

        result: dict[str, Any] = {
            "model_age_days": age_days,
            "staleness_threshold_days": self.model_staleness_days,
            "stale": age_days > self.model_staleness_days,
            "last_trained": mtime.isoformat(),
        }

        logger.info("Model staleness check complete", **result)
        return result

    def run_drift_checks(self) -> dict[str, Any]:
        """Run all drift checks."""
        logger.info("Starting drift detection checks")

        with Timer("drift_detection", logger):
            results: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "data_drift": {},
                "prediction_drift": {},
                "model_staleness": {},
                "overall_drift_detected": False,
            }

            self.reference_data = self.load_reference_data()
            self.current_data = self.load_current_data()

            if self.reference_data is not None and self.current_data is not None:
                results["data_drift"] = self.detect_data_drift(
                    self.reference_data, self.current_data
                )
                results["prediction_drift"] = self.detect_prediction_drift(self.current_data)
            else:
                logger.warning("Skipping drift checks - insufficient data")
                results["data_drift"] = {"error": "Insufficient data", "drift_detected": False}
                results["prediction_drift"] = {
                    "error": "Insufficient data",
                    "drift_detected": False,
                }

            results["model_staleness"] = self.check_model_staleness()

            data_drift_flag = bool(results["data_drift"].get("drift_detected", False))
            pred_drift_flag = bool(results["prediction_drift"].get("drift_detected", False))
            staleness_flag = bool(results["model_staleness"].get("stale", False))

            results["overall_drift_detected"] = data_drift_flag or pred_drift_flag or staleness_flag

            ensure_dir("reports/drift")
            save_json(results, "reports/drift/drift_report.json")

            logger.info("Drift detection complete", overall_drift=results["overall_drift_detected"])
            return results


def main() -> int:
    """Main drift detection entry point."""
    from mlops_fraud_detection.common.utils import setup_logging

    setup_logging("INFO")
    config = load_params()

    detector = DriftDetector(config)
    results = detector.run_drift_checks()

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

    if results["overall_drift_detected"]:
        print("\n⚠️  DRIFT DETECTED - Consider retraining")
        return 1
    else:
        print("\n✅ No significant drift detected")
        return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
