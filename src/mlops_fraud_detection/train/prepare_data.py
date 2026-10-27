"""Data preparation script for fraud detection pipeline.

This script:
1. Loads raw data
2. Performs train/val/test split
3. Handles missing values
4. Encodes categorical features
5. Scales numeric features
6. Saves processed datasets and preprocessor
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from mlops_fraud_detection.common.utils import (
    Timer,
    ensure_dir,
    get_logger,
    load_params,
    save_json,
)

logger = get_logger(__name__)


def load_raw_data(raw_path: str) -> pd.DataFrame:
    """Load raw fraud detection data."""
    logger.info("Loading raw data", path=raw_path)
    df = pd.read_csv(raw_path)
    logger.info("Raw data loaded", shape=df.shape, columns=list(df.columns))
    return df


def validate_data(df: pd.DataFrame, target_column: str) -> dict:
    """Validate data quality and return validation report."""
    logger.info("Validating data")

    report = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "target_distribution": df[target_column].value_counts().to_dict()
        if target_column in df.columns
        else {},
        "duplicate_rows": int(df.duplicated().sum()),
    }

    # Check for target column
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")

    # Log validation summary
    logger.info(
        "Data validation complete",
        **{k: v for k, v in report.items() if k not in ["missing_values", "dtypes"]},
    )

    return report


def split_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    val_size: float,
    stratify: bool,
    random_state: int,
) -> tuple:
    """Split data into train/val/test sets."""
    from sklearn.model_selection import train_test_split

    logger.info("Splitting data", test_size=test_size, val_size=val_size, stratify=stratify)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # First split: train+val vs test
    stratify_param = y if stratify else None
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    stratify_param = y_temp if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        stratify=stratify_param,
    )

    logger.info(
        "Data split complete",
        train_shape=X_train.shape,
        val_shape=X_val.shape,
        test_shape=X_test.shape,
        train_target_dist=y_train.value_counts().to_dict(),
        val_target_dist=y_val.value_counts().to_dict(),
        test_target_dist=y_test.value_counts().to_dict(),
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(
    X_train: pd.DataFrame, handle_missing: str, encode_categorical: str, scale_numeric: bool
):
    """Build sklearn preprocessing pipeline."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

    logger.info(
        "Building preprocessor",
        handle_missing=handle_missing,
        encode_categorical=encode_categorical,
        scale_numeric=scale_numeric,
    )

    # Identify column types
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    logger.info(
        "Feature types identified",
        n_numeric=len(numeric_features),
        n_categorical=len(categorical_features),
    )

    # Numeric transformer
    numeric_steps = []
    if handle_missing in ["median", "mean"]:
        strategy = handle_missing
        numeric_steps.append(("imputer", SimpleImputer(strategy=strategy)))
    elif handle_missing == "drop":
        # Handled separately
        pass

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(numeric_steps) if numeric_steps else "passthrough"

    # Categorical transformer
    if encode_categorical == "onehot":
        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
    elif encode_categorical == "target":
        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", TargetEncoder(smooth="auto")),
            ]
        )
    elif encode_categorical == "label":
        from sklearn.preprocessing import OrdinalEncoder

        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]
        )
    else:
        categorical_transformer = "passthrough"

    # Combine transformers
    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers, remainder="drop", verbose_feature_names_out=False
    )

    return preprocessor, numeric_features, categorical_features


def prepare_data(config: DictConfig) -> dict:
    """Main data preparation pipeline."""
    params = config

    with Timer("data_preparation", logger):
        # Load raw data
        df = load_raw_data(params.data.raw_path)

        # Validate
        validation_report = validate_data(df, params.data.target_column)

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df,
            target_column=params.data.target_column,
            test_size=params.preprocessing.test_size,
            val_size=params.preprocessing.val_size,
            stratify=params.preprocessing.stratify,
            random_state=params.training.random_state,
        )

        # Build and fit preprocessor
        preprocessor, numeric_features, categorical_features = build_preprocessor(
            X_train,
            handle_missing=params.preprocessing.handle_missing,
            encode_categorical=params.preprocessing.encode_categorical,
            scale_numeric=params.preprocessing.scale_numeric,
        )

        logger.info("Fitting preprocessor on training data")
        X_train_processed = preprocessor.fit_transform(X_train, y_train)
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)

        # Get feature names after transformation
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]

        # Create processed DataFrames
        train_df = pd.DataFrame(X_train_processed, columns=feature_names)
        train_df[params.data.target_column] = y_train.values

        val_df = pd.DataFrame(X_val_processed, columns=feature_names)
        val_df[params.data.target_column] = y_val.values

        test_df = pd.DataFrame(X_test_processed, columns=feature_names)
        test_df[params.data.target_column] = y_test.values

        # Save processed data
        ensure_dir(params.data.processed_dir)
        train_df.to_parquet(params.data.train_file, index=False)
        val_df.to_parquet(params.data.val_file, index=False)
        test_df.to_parquet(params.data.test_file, index=False)

        # Save preprocessor
        ensure_dir(Path(params.model.preprocessor_file).parent)
        joblib.dump(preprocessor, params.model.preprocessor_file)

        # Save feature info
        feature_info = {
            "feature_names": feature_names.tolist(),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "target_column": params.data.target_column,
            "n_features": len(feature_names),
        }
        save_json(feature_info, "data/processed/feature_info.json")

        # Save validation report
        ensure_dir("reports/data_validation")
        save_json(validation_report, "reports/data_validation/validation_report.json")

        logger.info(
            "Data preparation complete",
            train_path=params.data.train_file,
            val_path=params.data.val_file,
            test_path=params.data.test_file,
            preprocessor_path=params.model.preprocessor_file,
            n_features=len(feature_names),
        )

        return {
            "train_shape": train_df.shape,
            "val_shape": val_df.shape,
            "test_shape": test_df.shape,
            "n_features": len(feature_names),
            "feature_names": feature_names.tolist(),
        }


if __name__ == "__main__":
    from mlops_fraud_detection.common.utils import setup_logging

    setup_logging("INFO")
    params = load_params()
    result = prepare_data(params)
    print(f"Data preparation complete: {result}")
