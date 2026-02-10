#!/usr/bin/env python3
"""
Evaluation script for BMCI retrieval model on SatRain dataset.
All configuration is read from TOML files.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr

from satrain_models import (
    SatRainConfig,
    SatRainDataModule,
)
from satrain_models.bmci import BMCI

LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Path to model file (NetCDF format).")
args = parser.parse_args()


def create_retrieval_fn(model, satrain_config, datamodule):
    """
    Create retrieval function for BMCI model.
    """

    def retrieval_fn(input_data: xr.Dataset) -> xr.Dataset:
        """
        Run retrieval on input data.
        """
        # The BMCI model was trained on tabular data with features per sample.
        # For evaluation, we need to extract the same features from spatial data.

        # Extract features based on what was used in training (GMI channels)
        retrieval_input = datamodule._get_retrieval_input()
        features = []
        for inpt in retrieval_input:
            features += list(inpt.features.keys())
        inpt = np.concatenate([input_data[var].data for var in features], axis=1)
        pred = model.predict(inpt.T).squeeze()
        results = xr.Dataset({"surface_precip": (("batch",), pred)})
        return results

    return retrieval_fn


def main():
    """Main evaluation function - all configuration from TOML files."""

    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Provided model path '{model_path}' doesn't exist.")

    LOGGER.info(f"Loading BMCI model from {model_path}")
    bmci_model = BMCI.load(model_path)

    # Load dataset configuration from the model directory or use default
    dataset_config_path = model_path.parent / "dataset.toml"
    if not dataset_config_path.exists():
        dataset_config_path = Path("dataset.toml")
    
    if not dataset_config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")

    satrain_config = SatRainConfig.from_toml_file(dataset_config_path)

    # Disable angles for evaluation to match training
    if hasattr(satrain_config, "retrieval_input") and satrain_config.retrieval_input:
        satrain_config.retrieval_input[0].include_angles = False

    LOGGER.info(f"Loaded SatRain config from {dataset_config_path}")

    # Create data module
    data_module = SatRainDataModule(config=satrain_config)

    # Create experiment name
    experiment_name = model_path.stem

    # Run evaluation on different domains
    results = []
    domains = ["austria", "conus", "korea"]
    for domain in domains:
        LOGGER.info("Running evaluation for domain '%s'.", domain)
        evaluator = data_module.get_evaluator(domain)
        evaluator.evaluate(
            create_retrieval_fn(bmci_model, satrain_config, data_module),
            input_data_format="tabular",
            batch_size=8 * 2048,
        )
        results.append(evaluator.get_results())

    # Combine results
    results = xr.concat(results, dim="domain")
    results["domain"] = domains

    # Add model metadata
    model_metadata = {
        # Model identification
        "experiment_name": experiment_name,
        "model_type": "BMCI",
        # Model complexity metrics
        "num_samples": bmci_model.X.shape[0] if bmci_model.X is not None else 0,
        "num_features": bmci_model.X.shape[1] if bmci_model.X is not None else 0,
        "cutoff": bmci_model.cutoff,
        "sigma_mean": float(np.mean(np.sqrt(1.0 / bmci_model.Sinv))),
        "sigma_std": float(np.std(np.sqrt(1.0 / bmci_model.Sinv))),
    }

    results.attrs.update(model_metadata)

    # Save results
    output_path = Path(".") / "test_results"
    output_path.mkdir(exist_ok=True)
    results.to_netcdf(output_path / f"{experiment_name}.nc")

    LOGGER.info(f"Evaluation results saved to {output_path / f'{experiment_name}.nc'}")

    # Print summary statistics
    print("\n=== Evaluation Summary ===")
    for domain in domains:
        domain_results = results.sel(domain=domain)
        if "mse" in domain_results.data_vars:
            mse = float(domain_results.mse.values)
            print(f"{domain}: MSE = {mse:.6f}")
        if "correlation" in domain_results.data_vars:
            corr = float(domain_results.correlation.values)
            print(f"{domain}: Correlation = {corr:.6f}")


if __name__ == "__main__":
    main()