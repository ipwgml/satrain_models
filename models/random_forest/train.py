#!/usr/bin/env python3
"""
Training script for Random Forest retrieval model on SatRain dataset.
All configuration is read from TOML files.
"""

import argparse
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import tomli
import xarray as xr

from satrain_models import (
    SatRainConfig,
    SatRainDataModule,
    create_random_forest_retrieval,
)

LOGGER = logging.getLogger("random_forest_training")


class RandomForestTrainer:
    """Custom trainer for Random Forest models using SatRain tabular data."""

    def __init__(self, model, datamodule, config):
        self.model = model
        self.datamodule = datamodule
        self.config = config
        self.training_history = []

    def train(self):
        """Train the Random Forest model."""
        LOGGER.info("Loading training data in tabular format...")
        X_train, y_train = self.datamodule.load_tabular_data("training")

        LOGGER.info("Loading validation data in tabular format...")
        X_val, y_val = self.datamodule.load_tabular_data("validation")

        # Remove NaN values
        train_mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        val_mask = np.isfinite(X_val).all(axis=1) & np.isfinite(y_val)

        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_val = X_val[val_mask]
        y_val = y_val[val_mask]

        LOGGER.info(f"After filtering NaN - Train: {X_train.shape}, Val: {X_val.shape}")

        # Train the model
        LOGGER.info("Starting Random Forest training...")
        start_time = time.time()

        print("SHAPE :: ", X_train.shape)
        self.model.fit(X_train, y_train)

        print(X_train.shape, X_val.shape)

        training_time = time.time() - start_time
        LOGGER.info(f"Training completed in {training_time:.2f} seconds")

        # Calculate final metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        train_mse = np.mean((train_pred.squeeze() - y_train) ** 2)
        val_mse = np.mean((val_pred.squeeze() - y_val) ** 2)

        LOGGER.info(
            f"Final metrics - Train MSE: {train_mse:.6f}, Val MSE: {val_mse:.6f}"
        )

        return {
            "train_mse": float(train_mse),
            "val_mse": float(val_mse),
            "training_time": training_time,
            "num_parameters": self.model.num_parameters,
        }


def main():
    """Training function"""

    parser = argparse.ArgumentParser(description="Train Random Forest retrieval model")
    parser.add_argument(
        "compute_config", help="Path to compute config file.", default="compute.toml"
    )
    parser.add_argument(
        "--dataset-config", help="Path to dataset config file.", default="dataset.toml"
    )
    args = parser.parse_args()

    # Load dataset configuration
    dataset_config_path = Path(args.dataset_config)
    if not dataset_config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")
    config = SatRainConfig.from_toml_file(dataset_config_path)
    LOGGER.info(f"Loaded SatRain config from: {dataset_config_path}")

    # Load configuration files
    compute_config_path = Path(args.compute_config)
    if not compute_config_path.exists():
        raise FileNotFoundError(f"Compute config not found: {compute_config_path}")

    # Load full config to extract Random Forest section
    with open(compute_config_path, "rb") as f:
        full_config = tomli.load(f)
    rf_config = full_config.get("random_forest", {})

    LOGGER.info(f"Loaded compute config from: {compute_config_path}")
    LOGGER.info(f"Random Forest config: {rf_config}")

    # Create data module
    datamodule = SatRainDataModule(config=config)

    # Create model
    max_depth = rf_config.get("max_depth", None)
    if max_depth == "null":
        max_depth = None

    rf_model = create_random_forest_retrieval(
        n_estimators=rf_config.get("n_estimators", 100),
        max_depth=max_depth,
        min_samples_split=rf_config.get("min_samples_split", 2),
        min_samples_leaf=rf_config.get("min_samples_leaf", 1),
        max_features=rf_config.get("max_features", "sqrt"),
        bootstrap=rf_config.get("bootstrap", True),
        random_state=rf_config.get("random_state", 42),
        n_jobs=rf_config.get("n_jobs", -1),
    )

    LOGGER.info(f"Created Random Forest model")

    # Create trainer and train
    trainer = RandomForestTrainer(
        model=rf_model, datamodule=datamodule, config=rf_config
    )

    # Train the model
    LOGGER.info(f"Starting training with config: {compute_config_path}")
    results = trainer.train()

    # Save model
    LOGGER.info("Saving trained model...")
    output_path = Path("models")
    output_path.mkdir(exist_ok=True)

    # Create experiment name based on dataset configuration and Random Forest configuration
    dataset_prefix = config.get_experiment_name_prefix("random_forest")
    config_parts = [
        f"n{rf_config.get('n_estimators', 100)}",
        (
            f"d{rf_config.get('max_depth', 'inf')}"
            if rf_config.get("max_depth") is not None
            else "dinf"
        ),
        f"split{rf_config.get('min_samples_split', 2)}",
        f"leaf{rf_config.get('min_samples_leaf', 1)}",
        f"feat{rf_config.get('max_features', 'sqrt')}".replace(".", ""),
    ]

    base_name = f"{dataset_prefix}_{'_'.join(config_parts)}"

    # Find next available version number
    version = 0
    while True:
        experiment_name = f"{base_name}_v{version}"
        model_path = output_path / f"{experiment_name}.joblib"
        full_model_path = output_path / f"{experiment_name}.pkl"
        if not model_path.exists() and not full_model_path.exists():
            break
        version += 1

    LOGGER.info(f"Using experiment name: {experiment_name}")
    rf_model.save_model(str(model_path))

    # Save model with SatRain config using pickle
    with open(full_model_path, "wb") as f:
        pickle.dump(
            {
                "model_path": str(model_path),
                "satrain_config": config.to_dict(),
                "random_forest_config": rf_config,
                "training_results": results,
                "model": rf_model,
            },
            f,
        )

    LOGGER.info(f"Model saved to {model_path} and {full_model_path}")

    # Save training metrics to NetCDF
    LOGGER.info("Saving training metrics...")
    netcdf_dir = Path("netcdf_metrics")
    netcdf_dir.mkdir(exist_ok=True)

    # Create metrics dataset
    metrics_data = xr.Dataset(
        {
            "train_mse": (["epoch"], [results["train_mse"]]),
            "val_mse": (["epoch"], [results["val_mse"]]),
        },
        coords={"epoch": [0]},
    )

    metadata = {
        "experiment": dataset_prefix,
        "experiment_name": experiment_name,
        "version": f"v{version}",
        "model_type": "RandomForest",
        "num_parameters": results["num_parameters"],
        "training_time": results["training_time"],
    }

    # Convert rf_config values to NetCDF-compatible types
    for key, value in rf_config.items():
        if isinstance(value, bool):
            metadata[key] = str(value)
        else:
            metadata[key] = value

    metrics_data.attrs.update(metadata)

    output_path = netcdf_dir / f"{experiment_name}.nc"
    metrics_data.to_netcdf(output_path)
    LOGGER.info(f"Metrics saved to {output_path}")


if __name__ == "__main__":
    main()
