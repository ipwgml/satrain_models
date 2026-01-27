#!/usr/bin/env python3
"""
Clean PyTorch Lightning testing script for EncoderDecoder model on SatRain dataset.
All configuration is read from TOML files and saved model.
"""
import argparse
import logging
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
import xarray as xr

from satrain_models import (
    ComputeConfig,
    SatRainConfig,
    SatRainDataModule,
    SatRainEstimationModule,
)
from satrain_models.encoder_decoder import (
    EncoderDecoder,
    EncoderDecoderConfig,
    Conv2dBnReLU,
    Conv2dLnGELU,
    Conv2dLnReLU,
    ResNeXtBlock,
)

LOGGER = logging.getLogger(__name__)

# Available block classes mapping
BLOCK_CLASSES = {
    "Conv2dBnReLU": Conv2dBnReLU,
    "Conv2dLnGELU": Conv2dLnGELU,
    "Conv2dLnReLU": Conv2dLnReLU,
    "ResNeXtBlock": ResNeXtBlock,
}


def create_encoder_decoder_from_config(model_config: EncoderDecoderConfig, num_features: int) -> EncoderDecoder:
    """Create EncoderDecoder model from configuration."""
    block_class = BLOCK_CLASSES[model_config.block_class]
    
    return EncoderDecoder(
        block_class=block_class,
        in_channels=num_features,
        channels=model_config.channels,
        depths=model_config.depths,
        out_channels=model_config.out_channels,
        bilinear=model_config.bilinear,
    )


def main():
    """Main testing function - all configuration from TOML files and saved model."""
    parser = argparse.ArgumentParser(description="Test EncoderDecoder model")
    parser.add_argument("model", help="Path to model weight or checkpoint file.")
    args = parser.parse_args()

    # Load weights and configs
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Provided model path '{model_path}' doesn't exist.")

    loaded = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    state = loaded["state_dict"]

    # Remove model prefix for checkpoint files.
    if model_path.suffix == ".ckpt":
        state = {key[6:]: val for key, val in state.items()}

    satrain_config = SatRainConfig(**loaded["satrain_config"])
    LOGGER.info(f"Loaded SatRain config from model file {model_path}")
    
    # Load model configuration if available
    if "model_config" in loaded:
        model_config = EncoderDecoderConfig(**loaded["model_config"])
        LOGGER.info(f"Loaded model config from model file: {model_config.model_name}")
    else:
        # Fallback: try to load from separate TOML file
        model_config_path = Path("model.toml")
        if model_config_path.exists():
            model_config = EncoderDecoderConfig.from_toml_file(model_config_path)
            LOGGER.info(f"Loaded model config from: {model_config_path}")
        else:
            raise FileNotFoundError(
                "No model configuration found. Either save model with model_config "
                "or provide model.toml file."
            )

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
    encoder_decoder_model = create_encoder_decoder_from_config(
        model_config=model_config,
        num_features=satrain_config.num_features,
    )
    encoder_decoder_model.load_state_dict(state)
    
    LOGGER.info(f"Loaded {model_config.model_name} with {encoder_decoder_model.num_parameters:,} parameters")

    # Create Lightning module
    lightning_module = SatRainEstimationModule(
        model=encoder_decoder_model,
        loss=nn.MSELoss(),
        approach=compute_config.approach,
        name=model_config.model_name,
    )
    experiment_name = lightning_module.experiment_name

    # Run evaluation on different domains
    results = []
    domains = ["austria", "conus", "korea"]
    for domain in domains:
        LOGGER.info("Running evaluation for domain '%s'.", domain)
        evaluator = data_module.get_evaluator(domain)
        evaluator.evaluate(
            lightning_module.get_retrieval_fn(satrain_config, compute_config),
            batch_size=4 * compute_config.batch_size,
            tile_size=satrain_config.tile_size,
        )
        results.append(evaluator.get_results())

    results = xr.concat(results, dim="domain")
    results["domain"] = domains

    # Save results with model information
    output_path = Path(".") / "test_results"
    output_path.mkdir(exist_ok=True)
    

    # Add comprehensive model metadata to results
    model_metadata = {
        # Model identification
        "model_name": model_config.model_name,
        "experiment_name": experiment_name,
        
        # Model architecture details
        "block_class": model_config.block_class,
        "in_channels": model_config.in_channels,
        "out_channels": model_config.out_channels,
        "channels": str(model_config.channels),  # Convert list to string for NetCDF
        "depths": str(model_config.depths),      # Convert list to string for NetCDF
        "bilinear": int(model_config.bilinear),  # Convert boolean to int for NetCDF
        
        # Model complexity metrics
        "num_parameters": encoder_decoder_model.num_parameters,
        "num_trainable_parameters": encoder_decoder_model.num_trainable_parameters,
    }
    
    results.attrs.update(model_metadata)
    
    results.to_netcdf(output_path / f"{experiment_name}.nc")
    
    LOGGER.info(f"Evaluation completed. Results saved to: {output_path / f'{experiment_name}.nc'}")


if __name__ == "__main__":
    main()
