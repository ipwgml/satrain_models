#!/usr/bin/env python3
"""
Clean PyTorch Lightning training script for ResNet model on SatRain dataset.
All configuration is read from TOML files.

Created: January 2026
questions: veljko.petkovic@umd.edu
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import torch

import lightning as L
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from satrain_models import (
    ComputeConfig,
    SatRainConfig,
    SatRainDataModule,
    SatRainEstimationModule,
    create_resnet_from_config,
    ResNetConfig,
    tensorboard_to_netcdf,
)

# Override tile_size for ResNet model
# E.g., use 64x64 tiles which fit within GMI swath width (221 pixels)
RESNET_TILE_SIZE = (64, 64)
SatRainConfig.tile_size = property(lambda self: RESNET_TILE_SIZE)

LOGGER = logging.getLogger("resnet_training")

# veljko
# Default (highest precision, TensorFloat32 disabled)
#torch.set_float32_matmul_precision('highest')

# Faster, less precise (uses TensorFloat32)
torch.set_float32_matmul_precision('high')

# Even faster, even less precise (uses BFloat16)
#torch.set_float32_matmul_precision('medium')
#veljko


def save_training_metadata(
    output_path: Path,
    experiment_name: str,
    model_config: dict,
    dataset_config: dict,
    compute_config: dict,
    training_info: dict,
):
    """
    Save training metadata to a JSON file alongside the model.

    Args:
        output_path: Directory to save the metadata file
        experiment_name: Name of the experiment (used for filename)
        model_config: Model architecture parameters
        dataset_config: Dataset configuration
        compute_config: Compute/training configuration
        training_info: Additional training information
    """
    metadata = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        "model_architecture": model_config,
        "dataset_config": dataset_config,
        "compute_config": compute_config,
        "training_info": training_info,
    }

    metadata_path = output_path / f"{experiment_name}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    LOGGER.info(f"Saved training metadata to: {metadata_path}")


def main():
    """Training function"""

    parser = argparse.ArgumentParser(description="Train ResNet model")
    parser.add_argument("model_config", help="Path to model configuration TOML file")
    parser.add_argument(
        "--dataset-config", help="Path to dataset config file.", default="dataset.toml"
    )
    args = parser.parse_args()

    # Load model configuration
    model_config_path = Path(args.model_config)
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")
    model_config = ResNetConfig.from_toml_file(model_config_path)
    LOGGER.info(f"Loaded model config from: {model_config_path}")

    # Load dataset configuration
    dataset_config_path = Path(args.dataset_config)
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

    # Create model from configuration
    model_config.n_channels = datamodule.num_features
    resnet_model = create_resnet_from_config(
        model_config=model_config,
        num_features=datamodule.num_features,
    )
    
    LOGGER.info(f"Created {model_config.model_name} with {sum(1 for _ in resnet_model.parameters())} parameters")
    
    # Store model architecture info for metadata
    model_arch_config = {
        "n_channels": model_config.n_channels,
        "n_outputs": model_config.n_outputs,
        "blocks_per_layer": model_config.blocks_per_layer,
        "n_layers": model_config.n_layers,
        "base_channels": model_config.base_channels,
        "num_parameters": sum(p.numel() for p in resnet_model.parameters()),
        "num_trainable_parameters": sum(p.numel() for p in resnet_model.parameters() if p.requires_grad),
    }

    # Create experiment name that includes dataset and model configuration
    dataset_prefix = config.get_experiment_name_prefix("resnet")
    full_experiment_name = f"{dataset_prefix}_{model_config.model_name}"
    
    # Create Lightning module with custom name
    lightning_module = SatRainEstimationModule(
        model=resnet_model,
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

    # Extract and save training metrics (only on rank 0 to avoid file conflicts)
    if trainer.is_global_zero:
        LOGGER.info(f"Training finished. Saving metrics.")
        current_log_dir = loggers[0].log_dir
        netcdf_dir = Path("netcdf_metrics")
        netcdf_dir.mkdir(exist_ok=True)
        log_path = Path(current_log_dir)
        version_name = log_path.name
        metadata = {
            "experiment": "resnet",
            "model_name": model_config.model_name,
            "version": version_name,
            "approach": compute_config.approach,
            "num_parameters": sum(p.numel() for p in resnet_model.parameters()),
        }
        output_path = netcdf_dir / (experiment_name + ".nc")
        tensorboard_to_netcdf(
            log_dir=current_log_dir, output_path=output_path, metadata=metadata
        )

        # Save model with dataset config
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

        # Save training metadata to separate JSON file
        save_training_metadata(
            output_path=output_path,
            experiment_name=experiment_name,
            model_config=model_arch_config,
            dataset_config={
                "base_sensor": config.base_sensor,
                "geometry": config.geometry,
                "subset": config.subset,
                "format": config.format,
                "tile_size": RESNET_TILE_SIZE,
                "retrieval_input": str(config.retrieval_input),
                "target_config": str(config.target_config) if hasattr(config, 'target_config') else None,
            },
            compute_config={
                "accelerator": compute_config.accelerator,
                "devices": compute_config.devices,
                "precision": compute_config.precision,
                "max_epochs": compute_config.max_epochs,
                "batch_size": compute_config.batch_size,
                "approach": compute_config.approach,
                "num_workers": compute_config.num_workers,
                "pin_memory": compute_config.pin_memory,
                "persistent_workers": compute_config.persistent_workers,
                "accumulate_grad_batches": compute_config.accumulate_grad_batches,
                "log_every_n_steps": compute_config.log_every_n_steps,
                "check_val_every_n_epoch": compute_config.check_val_every_n_epoch,
            },
            training_info={
                "epochs_trained": trainer.current_epoch + 1,
                "global_step": trainer.global_step,
                "best_val_loss": trainer.callback_metrics.get("val/loss", None),
                "final_train_loss": trainer.callback_metrics.get("train/loss", None),
                "log_dir": str(current_log_dir),
            },
        )


if __name__ == "__main__":
    main()
