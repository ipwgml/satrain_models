#!/usr/bin/env python3
"""
Test script for ResNet model on SatRain dataset.
Evaluates a trained model on test domains and saves results.

Created: January 2026
questions: veljko.petkovic@umd.edu
"""

import argparse
import logging
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from satrain_models import (
    ComputeConfig,
    SatRainConfig,
    SatRainDataModule,
    SatRainEstimationModule,
    create_resnet,
    tensorboard_to_netcdf,
)

# (veljko) Override tile_size for ResNet model
# Use 64x64 tiles which fit within GMI swath width (221 pixels)
RESNET_TILE_SIZE = (64, 64)
SatRainConfig.tile_size = property(lambda self: RESNET_TILE_SIZE)

LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Path to model weight or checkpoint file.")
args = parser.parse_args()


def main():
    """Main testing function - evaluates model on all test domains."""

    # Load weights and SatRain config
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Provided model path '{model_path}' doesn't exist.")

    loaded = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    state = loaded["state_dict"]

    # Remove model prefix for checkpoint files.
    if model_path.suffix == ".ckpt":
        state = {key[6:]: val for key, val in state.items()}

    satrain_config = SatRainConfig(**loaded["satrain_config"])
    satrain_config.retrieval_input[0].include_angles = False
    print(satrain_config)
    LOGGER.info(f"Loaded SatRain config from model file %s", model_path)

    # Load compute configuration
    compute_config_path = Path("compute.toml")
    if not compute_config_path.exists():
        raise FileNotFoundError(f"Compute config not found: {compute_config_path}")
    compute_config = ComputeConfig.from_toml_file(compute_config_path)
    LOGGER.info(f"Loaded compute config from: {compute_config_path}")

    # Create data module
    data_module = SatRainDataModule(
        config=satrain_config,
        batch_size=compute_config.batch_size,
        num_workers=compute_config.num_workers,
        pin_memory=compute_config.pin_memory,
        persistent_workers=compute_config.persistent_workers,
    )

    # Auto-detect model architecture from state dict
    has_layer4 = any('layer4' in key for key in state.keys())
    n_layers = 4 if has_layer4 else 3
    layer1_blocks = len([k for k in state.keys() if k.startswith('layer1.') and '.conv1.' in k])
    blocks_per_layer = max(layer1_blocks, 2)
    LOGGER.info(f"Detected model architecture: n_layers={n_layers}, blocks_per_layer={blocks_per_layer}")

    # Create model with detected architecture
    resnet_model = create_resnet(
        n_channels=satrain_config.num_features,
        n_outputs=1,
        blocks_per_layer=blocks_per_layer,
        n_layers=n_layers,
    )
    resnet_model.load_state_dict(state)

    # Create Lightning module
    lightning_module = SatRainEstimationModule(
        model=resnet_model,
        loss=nn.MSELoss(),
        approach=compute_config.approach,
    )
    experiment_name = lightning_module.experiment_name

    results = []
    domains = ["austria", "conus", "korea"]
    for domain in domains:
        LOGGER.info("Running evaluation for domain '%s'.", domain)
        evaluator = data_module.get_evaluator(domain)
        evaluator.evaluate(
            lightning_module.get_retrieval_fn(satrain_config, compute_config),
            batch_size=compute_config.batch_size,
            tile_size=satrain_config.tile_size,
        )
        results.append(evaluator.get_results())

        # Collect matched predictions and targets from evaluate_scene
        # evaluate_scene returns both surface_precip (predictions) and surface_precip_ref (targets)
        # on the same grid - we extract matched pairs where both are finite
        pass  # Exploration complete - matched data extraction implemented in inference_resnet.py

    results = xr.concat(results, dim="domain")
    results["domain"] = domains

    model_metadata = {
        # Model identification
        "experiment_name": lightning_module.experiment_name,
        # Model complexity metrics
        "num_parameters": resnet_model.num_parameters if hasattr(resnet_model, 'num_parameters') else sum(p.numel() for p in resnet_model.parameters()),
        "num_trainable_parameters": resnet_model.num_trainable_parameters if hasattr(resnet_model, 'num_trainable_parameters') else sum(p.numel() for p in resnet_model.parameters() if p.requires_grad),
    }
    results.attrs.update(model_metadata)

    output_path = Path(".") / "test_results"
    output_path.mkdir(exist_ok=True)
    results.to_netcdf(output_path / f"{experiment_name}.nc")


if __name__ == "__main__":
    main()
