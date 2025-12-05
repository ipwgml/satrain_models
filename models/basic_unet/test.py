#!/usr/bin/env python3
"""
Clean PyTorch Lightning training script for UNet model on SatRain dataset.
All configuration is read from TOML files.
"""
import argparse
import logging
from pathlib import Path

import lightning as L
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
    create_unet,
    tensorboard_to_netcdf,
)

LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Path to model weight or checkpoint file.")
args = parser.parse_args()


def main():
    """Main training function - all configuration from TOML files."""

    # Load weights and SatRain config
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Provided model path '{path}' doesn't exist.")
        return 1

    loaded = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    state = loaded["state_dict"]

    # Remove model prefix for checkpoint files.
    if model_path.suffix == ".ckpt":
        state = {key[6:]: val for key, val in state.items()}

    satrain_config = SatRainConfig(**loaded["satrain_config"])
    satrain_config.retrieval_input[0].include_angles=False
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

    # Create model
    unet_model = create_unet(
        n_channels=satrain_config.num_features, n_outputs=1, bilinear=False
    )
    unet_model.load_state_dict(state)

    # Create Lightning module
    lightning_module = SatRainEstimationModule(
        model=unet_model,
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

    results = xr.concat(results, dim="domain")
    results["domain"] = domains

    model_metadata = {
        # Model identification
        "experiment_name": lightning_module.experiment_name,
        # Model complexity metrics
        "num_parameters": unet_model.num_parameters,
        "num_trainable_parameters": unet_model.num_trainable_parameters,
    }
    results.attrs.update(model_metadata)

    output_path = Path(".") / "test_results"
    output_path.mkdir(exist_ok=True)
    results.to_netcdf(output_path / f"{experiment_name}.nc")


if __name__ == "__main__":
    main()
