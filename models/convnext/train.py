#!/usr/bin/env python3
"""
Training script for ConvNeXt model on SatRain dataset.
All configuration is read from TOML files.
"""
import logging
from pathlib import Path

import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from satrain_models import (
    SatRainEstimationModule, SatRainConfig, create_convnext,
    SatRainDataModule, tensorboard_to_netcdf, ComputeConfig
)
# from convnext import create_convnext  # Import from the file above

LOGGER = logging.getLogger("convnext_training")

def main():
    """Main training function - all configuration from TOML files."""

    # Load dataset configuration
    dataset_config_path = Path("dataset.toml")
    if not dataset_config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")
    config = SatRainConfig.from_toml_file(dataset_config_path)
    LOGGER.info(f"Loaded SatRain config from: {dataset_config_path}")

    # Load compute configuration
    compute_config_path = Path("compute.toml")
    if not compute_config_path.exists():
        raise FileNotFoundError(f"Compute config not found: {compute_config_path}")
    compute_config = ComputeConfig.from_toml_file(compute_config_path)
    LOGGER.info(f"Loaded compute config from: {compute_config_path}")

    # Create data module
    datamodule = SatRainDataModule(
        config=config,
        batch_size=compute_config.batch_size,
        num_workers=compute_config.num_workers,
        pin_memory=compute_config.pin_memory,
        persistent_workers=compute_config.persistent_workers,
    )

    # Create ConvNeXt model
    model_size = 'tiny'  # Options: 'tiny', 'small', 'base', 'large'
    convnext_model = create_convnext(
        n_channels=datamodule.num_features,
        n_outputs=1,
        model_size=model_size,
        pretrained=False,  # Use ImageNet pretrained weights
    )

    # Create Lightning module
    lightning_module = SatRainEstimationModule(
        model=convnext_model,
        loss=nn.MSELoss(),
        approach=compute_config.approach,
    )
    experiment_name = lightning_module.experiment_name

    loggers = [TensorBoardLogger(save_dir="lightning_logs", name=f"{experiment_name}_convnext_{model_size}")]

    # Create trainer
    trainer = L.Trainer(
        max_epochs=compute_config.max_epochs,
        accelerator=compute_config.accelerator,
        devices=compute_config.devices,
        strategy="auto",
        precision=compute_config.precision,
        logger=loggers,
        callbacks=lightning_module.default_callbacks(),
        log_every_n_steps=compute_config.log_every_n_steps,
        check_val_every_n_epoch=compute_config.check_val_every_n_epoch,
        accumulate_grad_batches=compute_config.accumulate_grad_batches,
    )

    # Train the model
    LOGGER.info(f"Starting the training: {compute_config_path}")
    trainer.fit(lightning_module, datamodule)

    # Extract and save training metrics
    LOGGER.info(f"Training finished. Saving metrics.")
    current_log_dir = loggers[0].log_dir
    netcdf_dir = Path("netcdf_metrics")
    netcdf_dir.mkdir(exist_ok=True)
    log_path = Path(current_log_dir)
    version_name = log_path.name
    metadata = {
        "experiment": "convnext_regression",
        "version": version_name,
        "approach": compute_config.approach,
    }
    output_path = netcdf_dir / (f"{experiment_name}_convnext_{model_size}_{version_name}_metrics.nc")
    tensorboard_to_netcdf(
        log_dir=current_log_dir, output_path=output_path, metadata=metadata
    )

    # Save model with dataset config
    output_path = Path("models")
    output_path.mkdir(exist_ok=True)
    lightning_module.save(config, output_path)


if __name__ == "__main__":
    main()
