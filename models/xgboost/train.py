#!/usr/bin/env python3
"""
Training script for XGBoost retrieval model on SatRain dataset.
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
    create_xgboost,
)

LOGGER = logging.getLogger("xgboost_training")


class XGBoostTrainer:
    """Custom trainer for XGBoost models using SatRain tabular data."""

    def __init__(self, task, model, datamodule, config):
        self.task = task
        self.model = model
        self.datamodule = datamodule
        self.config = config
        self.training_history = []

    def train(self):
        """Train the XGBoost model."""
        LOGGER.info("Loading training data in tabular format...")
        X_train, y_train = self.datamodule.load_tabular_data("training", task=self.task)

        LOGGER.info("Loading validation data in tabular format...")
        X_val, y_val = self.datamodule.load_tabular_data("validation", task=self.task)

        # Remove NaN values
        train_mask = np.isfinite(X_train).all(axis=1) & np.isfinite(y_train)
        val_mask = np.isfinite(X_val).all(axis=1) & np.isfinite(y_val)

        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        X_val = X_val[val_mask]
        y_val = y_val[val_mask]

        LOGGER.info(f"After filtering NaN - Train: {X_train.shape}, Val: {X_val.shape}")

        # Train the model
        LOGGER.info("Starting XGBoost training...")
        start_time = time.time()

        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=self.config.get("early_stopping_rounds", 50),
            verbose=True,
        )

        training_time = time.time() - start_time
        LOGGER.info(f"Training completed in {training_time:.2f} seconds")

        # Calculate final metrics
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        return {
            "training_time": training_time,
            "num_parameters": self.model.num_parameters,
        }


def main():
    """Training function"""

    parser = argparse.ArgumentParser(description="Train XGBoost retrieval model")
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

    # Load full config to extract XGBoost section
    with open(compute_config_path, "rb") as f:
        full_config = tomli.load(f)
    xgb_config = full_config.get("xgboost", {})

    LOGGER.info(f"Loaded compute config from: {compute_config_path}")
    LOGGER.info(f"XGBoost config: {xgb_config}")

    # Create data module
    datamodule = SatRainDataModule(config=config)
    task = xgb_config.get("task", "precipitation_estimation")

    # Create model
    xgb_model = create_xgboost(
        task=task,
        n_estimators=xgb_config.get("n_estimators", 1000),
        max_depth=xgb_config.get("max_depth", 6),
        learning_rate=xgb_config.get("learning_rate", 0.1),
        subsample=xgb_config.get("subsample", 0.8),
        colsample_bytree=xgb_config.get("colsample_bytree", 0.8),
        reg_alpha=xgb_config.get("reg_alpha", 0),
        reg_lambda=xgb_config.get("reg_lambda", 1),
        random_state=xgb_config.get("random_state", 42),
    )

    LOGGER.info(f"Created XGBoost model")

    # Create trainer and train
    trainer = XGBoostTrainer(
        task=task,
        model=xgb_model,
        datamodule=datamodule,
        config=xgb_config
    )

    # Train the model
    LOGGER.info(f"Starting training with config: {compute_config_path}")
    results = trainer.train()

    # Save model
    LOGGER.info("Saving trained model...")
    output_path = Path("models")
    output_path.mkdir(exist_ok=True)

    # Create experiment name based on dataset configuration and XGBoost configuration
    if task == "precipitation_estimation":
        dataset_prefix = config.get_experiment_name_prefix("xgboost_")
    else:
        dataset_prefix = config.get_experiment_name_prefix(f"xgboost_{task}")

    config_parts = [
        f"n{xgb_config.get('n_estimators', 1000)}",
        f"d{xgb_config.get('max_depth', 6)}",
        f"lr{xgb_config.get('learning_rate', 0.1):.3f}".replace(".", ""),
        f"sub{xgb_config.get('subsample', 0.8):.2f}".replace(".", ""),
        f"col{xgb_config.get('colsample_bytree', 0.8):.2f}".replace(".", ""),
    ]
    if xgb_config.get("reg_alpha", 0) > 0:
        config_parts.append(f"a{xgb_config.get('reg_alpha'):.3f}".replace(".", ""))
    if xgb_config.get("reg_lambda", 1) != 1:
        config_parts.append(f"l{xgb_config.get('reg_lambda'):.3f}".replace(".", ""))

    base_name = f"{dataset_prefix}_{'_'.join(config_parts)}"

    # Find next available version number
    version = 0
    while True:
        experiment_name = f"{base_name}_v{version}"
        model_path = output_path / f"{experiment_name}.xgb"
        full_model_path = output_path / f"{experiment_name}.pkl"
        if not model_path.exists() and not full_model_path.exists():
            break
        version += 1

    LOGGER.info(f"Using experiment name: {experiment_name}")
    xgb_model.save_model(str(model_path))

    # Save model with SatRain config using pickle
    with open(full_model_path, "wb") as f:
        pickle.dump(
            {
                "model_path": str(model_path),
                "satrain_config": config.to_dict(),
                "xgboost_config": xgb_config,
                "training_results": results,
                "model": xgb_model,
            },
            f,
        )

    LOGGER.info(f"Model saved to {model_path} and {full_model_path}")


if __name__ == "__main__":
    main()
