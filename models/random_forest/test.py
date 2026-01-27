#!/usr/bin/env python3
"""
Evaluation script for Random Forest retrieval model on SatRain dataset.
All configuration is read from TOML files.
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import tomli
import xarray as xr

from satrain_models import (
    SatRainConfig,
    SatRainDataModule,
    create_random_forest_retrieval,
)

LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Path to model weight or checkpoint file.")
args = parser.parse_args()


def create_retrieval_fn(model, satrain_config, datamodule):
    """
    Create retrieval function for Random Forest model.
    """

    def retrieval_fn(input_data: xr.Dataset) -> xr.Dataset:
        """
        Run retrieval on input data.
        """
        # The Random Forest model was trained on tabular data with features per sample.
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

    # Load model and SatRain config
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Provided model path '{model_path}' doesn't exist.")

    # Try loading as pickle first, then fallback to other formats
    if model_path.suffix == ".pkl":
        with open(model_path, "rb") as f:
            loaded = pickle.load(f)

        satrain_config = SatRainConfig(**loaded["satrain_config"])
        rf_config = loaded.get("random_forest_config", {})
        rf_model = loaded.get("model")
    else:
        # Handle other formats (legacy support)
        try:
            with open(model_path, "rb") as f:
                loaded = pickle.load(f)
        except:
            # Fallback - create new model and load from .joblib file
            satrain_config = SatRainConfig.from_toml_file("dataset.toml")
            rf_config = {}
            rf_model = create_random_forest_retrieval()
            rf_model_path = (
                str(model_path).replace(".pt", ".joblib").replace(".pkl", ".joblib")
            )
            if Path(rf_model_path).exists():
                rf_model.load_model(rf_model_path)
            else:
                raise FileNotFoundError(
                    f"Random Forest model file not found: {rf_model_path}"
                )

    # Disable angles for evaluation to match training
    if hasattr(satrain_config, "retrieval_input") and satrain_config.retrieval_input:
        satrain_config.retrieval_input[0].include_angles = False

    LOGGER.info(f"Loaded SatRain config from model file %s", model_path)

    # Create data module
    data_module = SatRainDataModule(config=satrain_config)

    # Create experiment name
    experiment_name = model_path.stem

    # Run evaluation on all domains using tabular format
    results = []
    domains = ["austria", "conus", "korea"]
    for domain in domains:
        LOGGER.info("Running evaluation for domain '%s'.", domain)
        evaluator = data_module.get_evaluator(domain)
        evaluator.evaluate(
            create_retrieval_fn(rf_model, satrain_config, data_module),
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
        "model_type": "RandomForest",
        # Model complexity metrics
        "num_parameters": rf_model.num_parameters,
        "num_trainable_parameters": rf_model.num_trainable_parameters,
    }

    # Add Random Forest configuration
    model_metadata.update(rf_config)
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
