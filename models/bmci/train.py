#!/usr/bin/env python3
"""
Training script for BMCI retrieval model on SatRain dataset.
All configuration is read from TOML files.
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import tomli
import xarray as xr

from satrain_models import (
    SatRainConfig,
    SatRainDataModule,
)
from satrain_models.bmci_fast import BMCIc

LOGGER = logging.getLogger("bmci_training")


class BMCITrainer:
    """Custom trainer for BMCI models using SatRain tabular data."""

    def __init__(self, sigma, cutoff, datamodule, config):
        self.sigma = sigma
        self.cutoff = cutoff
        self.datamodule = datamodule
        self.config = config

    def train(self):
        """Train the BMCI model."""
        LOGGER.info("Loading training data in tabular format...")
        X_train, y_train = self.datamodule.load_tabular_data("training")

        # Remove NaN values
        train_mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]

        LOGGER.info(f"After filtering NaN - Train: {X_train.shape}")

        # Create and train the model
        LOGGER.info("Creating BMCI model...")
        model = BMCIc(sigma=self.sigma, cutoff=self.cutoff)
        
        LOGGER.info("Starting BMCI training...")
        start_time = time.time()
        
        model.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        LOGGER.info(f"Training completed in {training_time:.2f} seconds")

        # Calculate final metrics on training data (since BMCI doesn't use validation during training)
        X_val, y_val = self.datamodule.load_tabular_data("validation")
        val_pred = model.predict(X_val)
        val_mse = np.mean((val_pred - y_val) ** 2)

        LOGGER.info(f"Validation MSE: {val_mse:.6f}")

        return {
            "val_mse": float(val_mse),
            "num_samples": len(X_train),
            "num_features": X_train.shape[1],
        }, model


def main():
    """Training function"""

    parser = argparse.ArgumentParser(description="Train BMCI retrieval model")
    parser.add_argument(
        "bmci_config", help="Path to BMCI config file.", default="bmci.toml"
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

    # Load BMCI configuration
    bmci_config_path = Path(args.bmci_config)
    if not bmci_config_path.exists():
        raise FileNotFoundError(f"BMCI config not found: {bmci_config_path}")

    with open(bmci_config_path, "rb") as f:
        bmci_config = tomli.load(f)

    LOGGER.info(f"Loaded BMCI config from: {bmci_config_path}")
    LOGGER.info(f"BMCI config: {bmci_config}")

    # Extract sigma and cutoff from config
    sigma = np.array(bmci_config["sigma"])
    cutoff = bmci_config.get("cutoff", None)

    LOGGER.info(f"Using sigma: {sigma}")
    LOGGER.info(f"Using cutoff: {cutoff}")

    # Create data module
    datamodule = SatRainDataModule(config=config)

    # Create trainer and train
    trainer = BMCITrainer(sigma=sigma, cutoff=cutoff, datamodule=datamodule, config=bmci_config)

    # Train the model
    LOGGER.info(f"Starting training with config: {bmci_config_path}")
    results, model = trainer.train()

    # Save model
    LOGGER.info("Saving trained model...")
    output_path = Path("models")
    output_path.mkdir(exist_ok=True)

    # Create experiment name based on dataset configuration and BMCI configuration
    dataset_prefix = config.get_experiment_name_prefix("bmci")
    config_parts = []
    
    # Add sigma description (use mean or first few values)
    if len(sigma) <= 3:
        sigma_desc = "_".join([f"{s:.3f}".replace(".", "") for s in sigma])
    else:
        sigma_desc = f"mean{np.mean(sigma):.3f}".replace(".", "")
    config_parts.append(f"s{sigma_desc}")
    
    # Add cutoff if specified
    if cutoff is not None:
        config_parts.append(f"c{cutoff:.3f}".replace(".", ""))

    base_name = f"{dataset_prefix}_{'_'.join(config_parts)}"

    # Find next available version number
    version = 0
    while True:
        experiment_name = f"{base_name}_v{version}"
        model_path = output_path / f"{experiment_name}.nc"
        if not model_path.exists():
            break
        version += 1

    LOGGER.info(f"Using experiment name: {experiment_name}")
    
    # Save model using BMCI's built-in save method
    model.save(model_path)

    LOGGER.info(f"Model saved to {model_path}")

    # Save training metrics to NetCDF
    LOGGER.info("Saving training metrics...")
    netcdf_dir = Path("netcdf_metrics")
    netcdf_dir.mkdir(exist_ok=True)

    # Create metrics dataset
    metrics_data = xr.Dataset(
        {
            "val_mse": (["epoch"], [results["val_mse"]]),
        },
        coords={"epoch": [0]},
    )

    metadata = {
        "experiment": dataset_prefix,
        "experiment_name": experiment_name,
        "version": f"v{version}",
        "model_type": "BMCI",
        "num_samples": results["num_samples"],
        "num_features": results["num_features"],
        "cutoff": str(cutoff),
    }
    metrics_data.attrs.update(metadata)

    metrics_output_path = netcdf_dir / f"{experiment_name}.nc"
    metrics_data.to_netcdf(metrics_output_path)
    LOGGER.info(f"Metrics saved to {metrics_output_path}")


if __name__ == "__main__":
    main()
