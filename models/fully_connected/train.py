#!/usr/bin/env python3
"""
Clean PyTorch Lightning training script for Fully-Connected model on SatRain dataset.
All configuration is read from TOML files.
"""
import argparse
import logging
from pathlib import Path

import lightning as L
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

import satrain_models
print(satrain_models, getattr(satrain_models, "__file__", None))

from satrain_models import (
    ComputeConfig,
    SatRainConfig,
    SatRainDataModule,
    SatRainEstimationModule,
    create_fully_connected,
    tensorboard_to_netcdf,
)

LOGGER = logging.getLogger("basic_fully_connected_training")


def main():
    """Training function"""
     
    parser = argparse.ArgumentParser(description="Train basic Fully Connected model")
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

    # Load compute configuration
    compute_config_path = Path(args.compute_config)
    if not compute_config_path.exists():
        raise FileNotFoundError(f"Compute config not found: {compute_config_path}")
    compute_config = ComputeConfig.from_toml_file(compute_config_path)
    LOGGER.info(f"Loaded compute config from: {compute_config_path}")

    # if 'fully_connected' not in compute_config.model_config:
    #     compute_config.model_config['fully_connected'] = {'hidden_layer': [8,2]}

    # for cfg in compute_config.model_config['fully_connected']:
    #     if 'hidden_layer' in cfg:
    #         hidden_layer = compute_config.model_config['fully_connected']['hidden_layer']

    # Create data module
    datamodule = SatRainDataModule(
        config=config,
        batch_size=compute_config.batch_size,
        num_workers=compute_config.num_workers,
        pin_memory=compute_config.pin_memory,
        persistent_workers=compute_config.persistent_workers,
    )
    
    # Create model
    fully_connected_model = create_fully_connected(
        input_dim=datamodule.num_features, hidden_dims=config.hidden_layer)
    
    # Create Lightning module with dataset-aware naming
    experiment_prefix = config.get_experiment_name_prefix("fully_connected")
    lightning_module = SatRainEstimationModule(
        model=fully_connected_model,
        loss=nn.MSELoss(),
        approach=compute_config.approach,
        name=experiment_prefix,
    )
    experiment_name = lightning_module.experiment_name

    loggers = [TensorBoardLogger(save_dir="lightning_logs", name=experiment_name)]

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
        "experiment": experiment_prefix,
        "version": version_name,
        "approach": compute_config.approach,
    }
    output_path = netcdf_dir / (experiment_name + ".nc")
    tensorboard_to_netcdf(
        log_dir=current_log_dir, output_path=output_path, metadata=metadata
    )

    # Save model with dataset config
    output_path = Path("models")
    output_path.mkdir(exist_ok=True)
    lightning_module.save(config, output_path)


if __name__ == "__main__":
    main()
