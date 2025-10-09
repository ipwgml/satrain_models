#!/usr/bin/env python3
"""
Clean PyTorch Lightning training script for UNet model on SatRain dataset.
All configuration is read from TOML files.
"""
import logging
from pathlib import Path

import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from satrain_models import (
    SatRainEstimationModule, create_unet, SatRainConfig, SatRainDataModule,
    tensorboard_to_netcdf
)


LOGGER = logging.getLogger("basic_unet_training")


def main():
    """Main training function - all configuration from TOML files."""
    


    # Load dataset configuration
    dataset_config_path = Path("dataset.toml")
    if not dataset_config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")
    config = SatRainConfig.from_toml_file(dataset_config_path)
    LOGGER.info(f"Loaded dataset config from: {dataset_config_path}")
    
    # Load compute configuration
    compute_config_path = Path("compute.toml")
    if not compute_config_path.exists():
        raise FileNotFoundError(f"Compute config not found: {compute_config_path}")
    compute_config = SatRainConfig.from_toml_file(compute_config_path)
    LOGGER.info(f"Loaded compute config from: {compute_config_path}")


    # Training settings
    compute_settings = getattr(compute_config, 'compute', {})
    max_epochs = compute_settings.get('max_epochs', 100)
    batch_size = compute_settings.get('batch_size', 8)
    num_workers = compute_settings.get('num_workers', 4)
    approach = compute_settings.get('approach', 'adamw_simple')
    
    # Hardware settings
    accelerator = compute_settings.get('accelerator', 'gpu')
    devices = compute_settings.get('devices', 1)
    precision = compute_settings.get('precision', '32')
    
    # Logging settings
    output_dir = compute_settings.get('output_dir', './lightning_logs')

    # Create data module
    datamodule = SatRainDataModule(
        config=config,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=compute_settings.get('pin_memory', True),
        persistent_workers=compute_settings.get('persistent_workers', True),
        spatial=True  # Use spatial dataset for CNN
    )

    # Create model
    unet_model = create_unet(
        n_channels=datamodule.num_features, 
        n_outputs=1, 
        bilinear=False
    )

    # Create Lightning module
    lightning_module = SatRainEstimationModule(
        model=unet_model,
        loss=nn.MSELoss(),
        approach=approach,
    )
    
    loggers = [TensorBoardLogger(save_dir=output_dir, name="basic_unet")]

    # Create trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy="auto",
        precision=precision,
        logger=loggers,
        callbacks=lightning_module.default_callbacks(),
        log_every_n_steps=compute_settings.get('log_every_n_steps', 10),
        check_val_every_n_epoch=compute_settings.get('check_val_every_n_epoch', 1),
        accumulate_grad_batches=compute_settings.get('accumulate_grad_batches', 1),
    )
    

    LOGGER.info(f"Starting the training: {compute_config_path}")
    # Train the model
    trainer.fit(lightning_module, datamodule)

    LOGGER.info(f"Training finished. Saving metrics.")
    # Save metrics to .netcdf
    current_log_dir = loggers[0].log_dir
    netcdf_dir = Path("netcdf_metrics")
    netcdf_dir.mkdir(exist_ok=True)
    log_path = Path(current_log_dir)
    version_name = log_path.name
    output_path = netcdf_dir / f"basic_unet_{version_name}_metrics.nc"
    metadata = {
        'experiment': 'basic_unet',
        'version': version_name,
        'approach': approach,
    }

    # Extract and save metrics
    tensorboard_to_netcdf(
        log_dir=current_log_dir,
        output_path=output_path,
        metadata=metadata
    )


if __name__ == '__main__':
    main()
