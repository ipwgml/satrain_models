#!/usr/bin/env python3
"""
Clean PyTorch Lightning training script for Fully-connected model on SatRain dataset.
All configuration is read from TOML files.
"""
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import xarray as xr

from satrain_models import (
    SatRainEstimationModule, ComputeConfig, create_fully_connected, SatRainConfig, SatRainDataModule,
    tensorboard_to_netcdf
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
        raise FileNotFoundError(
            f"Provided model path '{model_path}' doesn't exist."
        )
        return 1


    loaded = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    state = loaded["state_dict"]

    # Remove model prefix for checkpoint files.
    if model_path.suffix == ".ckpt":
        state = {key[6:]: val for key, val in state.items()}

    # satrain_config = SatRainConfig(**loaded["satrain_config"])
    # LOGGER.info(f"Loaded SatRain config from model file %s", model_path)

    satrain_config = SatRainConfig(**loaded["satrain_config"])
    #satrain_config.retrieval_input[0].include_angles = False
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
    fully_connected_model = create_fully_connected(
        input_dim=satrain_config.num_features,
        hidden_dims=satrain_config.hidden_layer)
    fully_connected_model.load_state_dict(state)

    # Create Lightning module
    lightning_module = SatRainEstimationModule(
        model=fully_connected_model,
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
            input_data_format="tabular",
        )
        results.append(evaluator.get_results())

    results = xr.concat(results, dim="domain")
    results["domain"] = domains

    output_path = Path(".") / "test_results"
    output_path.mkdir(exist_ok=True)
    results.to_netcdf(output_path / f"{experiment_name}.nc")


if __name__ == '__main__':
    main()
