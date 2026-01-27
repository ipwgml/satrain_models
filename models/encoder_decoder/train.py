#!/usr/bin/env python3
"""
Clean PyTorch Lightning training script for EncoderDecoder model on SatRain dataset.
All configuration is read from TOML files.
"""
import argparse
import logging
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.loggers import TensorBoardLogger

from satrain_models import (
    ComputeConfig,
    SatRainConfig,
    SatRainDataModule,
    SatRainEstimationModule,
    tensorboard_to_netcdf,
)
from satrain_models.encoder_decoder import (
    EncoderDecoder,
    EncoderDecoderConfig,
    Conv2dBnReLU,
    Conv2dLnGELU,
    Conv2dLnReLU,
    ResNeXtBlock,
    InvertedBottleneck
)

LOGGER = logging.getLogger("encoder_decoder_training")

# Available block classes mapping
BLOCK_FACTORIES = {
    "Conv2dBnReLU": Conv2dBnReLU,
    "Conv2dLnGELU": Conv2dLnGELU,
    "Conv2dLnReLU": Conv2dLnReLU,
    "ResNeXtBlock": ResNeXtBlock,
    "InvertedBottleneck": InvertedBottleneck,
}


def create_encoder_decoder_from_config(model_config: EncoderDecoderConfig, num_features: int) -> EncoderDecoder:
    """Create EncoderDecoder model from configuration."""
    block_factory = BLOCK_FACTORIES[model_config.block_factory]
    
    return EncoderDecoder(
        block_factory=block_factory,
        in_channels=num_features,
        channels=model_config.channels,
        depths=model_config.depths,
        out_channels=model_config.out_channels,
        bilinear=model_config.bilinear,
    )


def main():
    """Main training function - all configuration from TOML files."""
    parser = argparse.ArgumentParser(description="Train EncoderDecoder model")
    parser.add_argument("model_config", help="Path to model configuration TOML file")
    parser.add_argument(
        "--dataset-config", help="Path to dataset config file.", default="dataset.toml"
    )
    args = parser.parse_args()

    # Load model configuration
    model_config_path = Path(args.model_config)
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")
    model_config = EncoderDecoderConfig.from_toml_file(model_config_path)
    LOGGER.info(f"Loaded model config from: {model_config_path}")

    # Load dataset configuration
    dataset_config_path = Path(args.dataset_config)
    if not dataset_config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")
    config = SatRainConfig.from_toml_file(dataset_config_path)
    LOGGER.info(f"Loaded SatRain config from: {dataset_config_path}")
    model_config.in_channels = config.num_features

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

    # Create model
    encoder_decoder_model = create_encoder_decoder_from_config(
        model_config=model_config,
        num_features=datamodule.num_features,
    )
    
    LOGGER.info(f"Created {model_config.model_name} with {encoder_decoder_model.num_parameters:,} parameters")

    # Create experiment name that includes dataset configuration
    dataset_prefix = config.get_experiment_name_prefix("encoder_decoder")
    full_experiment_name = f"{dataset_prefix}_{model_config.model_name}"
    
    # Create Lightning module with custom name
    lightning_module = SatRainEstimationModule(
        model=encoder_decoder_model,
        loss=nn.MSELoss(),
        approach=compute_config.approach,
        name=full_experiment_name,
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
    LOGGER.info(f"Starting training: {model_config.model_name}")
    trainer.fit(lightning_module, datamodule)

    # Extract and save training metrics
    LOGGER.info(f"Training finished. Saving metrics.")
    current_log_dir = loggers[0].log_dir
    netcdf_dir = Path("netcdf_metrics")
    netcdf_dir.mkdir(exist_ok=True)
    log_path = Path(current_log_dir)
    version_name = log_path.name
    metadata = {
        "experiment": "encoder_decoder",
        "model_name": model_config.model_name,
        "version": version_name,
        "approach": compute_config.approach,
        "block_factory": model_config.block_factory,
        "num_parameters": encoder_decoder_model.num_parameters,
    }
    output_path = netcdf_dir / (experiment_name + ".nc")
    tensorboard_to_netcdf(
        log_dir=current_log_dir, output_path=output_path, metadata=metadata
    )

    # Save model with dataset and model config
    output_path = Path("models")
    output_path.mkdir(exist_ok=True)
    
    # Save model manually to include both configs
    model_path = output_path / f"{experiment_name}.pt"
    state = lightning_module.model.state_dict()
    torch.save({
        "state_dict": state,
        "satrain_config": config.to_dict(),
        "model_config": model_config.to_dict(),
    }, model_path)
    
    LOGGER.info(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
