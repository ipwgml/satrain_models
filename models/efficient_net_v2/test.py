#!/usr/bin/env python3

#!/usr/bin/env python3
"""
Clean PyTorch Lightning evaluation script for EfficientNetV2 model on SatRain dataset.
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
    SatRainEstimationModule, ComputeConfig, create_efficient_net_v2, SatRainConfig, SatRainDataModule,
    tensorboard_to_netcdf
)
LOGGER = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Path to model weight or checkpoint file (.pt, .pth, .ckpt)")
args = parser.parse_args()


def main():
    """Main evaluation function - all configuration from TOML files."""

    # Load weights and SatRain config
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Provided model path '{model_path}' doesn't exist."
        )

    # Check file extension
    if model_path.suffix == '.nc':
        raise ValueError(
            f"You provided a NetCDF metrics file (.nc). "
            f"Please provide a PyTorch checkpoint file (.pt, .pth, or .ckpt). "
            f"Look in the 'models/' or 'lightning_logs/' directory for checkpoint files."
        )

    # Load checkpoint/state_dict (accept multiple formats)
    loaded = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)

    # If checkpoint is a dict containing 'state_dict' (Lightning style), use it.
    if isinstance(loaded, dict) and "state_dict" in loaded:
        state = loaded["state_dict"]
        satrain_cfg = loaded.get("satrain_config", None)
    else:
        # Assume loaded is a bare state_dict
        state = loaded
        satrain_cfg = None

    # Remove common 'model.' or 'model_' prefixes from Lightning checkpoints
    if isinstance(state, dict) and any(k.startswith("model.") or k.startswith("model_") for k in state.keys()):
        state = {
            (k.split(".", 1)[1] if k.startswith("model.") else (k.split("_", 1)[1] if k.startswith("model_") else k)): v
            for k, v in state.items()
        }

    # Build SatRainConfig: prefer checkpoint's config, otherwise fallback to local dataset.toml
    if satrain_cfg is not None:
        satrain_config = SatRainConfig(**satrain_cfg)
    else:
        dataset_config_path = Path("dataset.toml")
        if not dataset_config_path.exists():
            raise FileNotFoundError(
                "Checkpoint has no 'satrain_config' and dataset.toml not found in working directory."
            )
        satrain_config = SatRainConfig.from_toml_file(dataset_config_path)
        LOGGER.info("Loaded SatRain config from dataset.toml (fallback)")

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

    # Create EfficientNetV2 model
    model_size = 'small'  # Options: 'small', 'medium', 'large'
    efficient_net_model = create_efficient_net_v2(
        n_channels=satrain_config.num_features,
        n_outputs=1,
        model_size=model_size,
        pretrained=False,  # we can use pretrained weights if needed
    )
    efficient_net_model.load_state_dict(state)
    
    # Create Lightning module
    lightning_module = SatRainEstimationModule(
        model=efficient_net_model,
        loss=nn.MSELoss(),
        approach=compute_config.approach,
    )
    experiment_name = f"{lightning_module.experiment_name}_efficient_net_v2_{model_size}"

    results = []
    domains = ["austria"]  # , "conus", "korea"]
    for domain in domains:
        LOGGER.info("Running evaluation for domain '%s'.", domain)
        evaluator = data_module.get_evaluator(domain)
        evaluator.evaluate(
            lightning_module.get_retrieval_fn(satrain_config, compute_config),
            batch_size=compute_config.batch_size,
            tile_size=(64, 64)
        )
        results.append(evaluator.get_results())

    results = xr.concat(results, dim="domain")
    results["domain"] = domains

    output_path = Path(".") / "test_results"
    output_path.mkdir(exist_ok=True)
    results.to_netcdf(output_path / f"{experiment_name}.nc")


if __name__ == '__main__':
    main()

# """
# Clean PyTorch Lightning evaluation script for EfficientNetV2 model on SatRain dataset.
# All configuration is read from TOML files.
# """
# import argparse
# import logging
# from pathlib import Path

# import torch
# import torch.nn as nn
# import lightning as L
# from lightning.pytorch.loggers import TensorBoardLogger
# from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
# import xarray as xr


# from satrain_models import (
#     SatRainEstimationModule, ComputeConfig, create_efficient_net_v2, SatRainConfig, SatRainDataModule,
#     tensorboard_to_netcdf
# )
# LOGGER = logging.getLogger(__name__)

# parser = argparse.ArgumentParser()
# parser.add_argument("model", help="Path to model weight or checkpoint file.")
# args = parser.parse_args()


# def main():
#     """Main training function - all configuration from TOML files."""

#     # Load weights and SatRain config
#     model_path = Path(args.model)
#     if not model_path.exists():
#         raise FileNotFoundError(
#             f"Provided model path '{model_path}' doesn't exist."
#         )
#         return 1

#     # Load checkpoint/state_dict (accept multiple formats)
#     loaded = torch.load(model_path, map_location=torch.device("cpu"))

#     # If checkpoint is a dict containing 'state_dict' (Lightning style), use it.
#     if isinstance(loaded, dict) and "state_dict" in loaded:
#         state = loaded["state_dict"]
#         satrain_cfg = loaded.get("satrain_config", None)
#     else:
#         # Assume loaded is a bare state_dict
#         state = loaded
#         satrain_cfg = None

#     # Remove common 'model.' or 'model_' prefixes from Lightning checkpoints
#     if isinstance(state, dict) and any(k.startswith("model.") or k.startswith("model_") for k in state.keys()):
#         state = {
#             (k.split(".", 1)[1] if k.startswith("model.") else (k.split("_", 1)[1] if k.startswith("model_") else k)): v
#             for k, v in state.items()
#         }

#     # Build SatRainConfig: prefer checkpoint's config, otherwise fallback to local dataset.toml
#     if satrain_cfg is not None:
#         satrain_config = SatRainConfig(**satrain_cfg)
#     else:
#         dataset_config_path = Path("dataset.toml")
#         if not dataset_config_path.exists():
#             raise FileNotFoundError(
#                 "Checkpoint has no 'satrain_config' and dataset.toml not found in working directory."
#             )
#         satrain_config = SatRainConfig.from_toml_file(dataset_config_path)
#         LOGGER.info("Loaded SatRain config from dataset.toml (fallback)")

#     # Load compute configuration
#     compute_config_path = Path("compute.toml")
#     if not compute_config_path.exists():
#         raise FileNotFoundError(f"Compute config not found: {compute_config_path}")
#     compute_config = ComputeConfig.from_toml_file(compute_config_path)
#     LOGGER.info(f"Loaded compute config from: {compute_config_path}")

#     # Create data module
#     data_module = SatRainDataModule(
#         config=satrain_config,
#         batch_size=compute_config.batch_size,
#         num_workers=compute_config.num_workers,
#         pin_memory=compute_config.pin_memory,
#         persistent_workers=compute_config.persistent_workers,
#     )

# # Create EfficientNetV2 model
#     model_size = 'small'  # Options: 'small', 'medium', 'large'
#     efficient_net_model = create_efficient_net_v2(
#         n_channels=satrain_config.num_features,
#         n_outputs=1,
#         model_size=model_size,
#         pretrained=False,  # we can use pretrained weights if needed
#     )
#     efficient_net_model.load_state_dict(state)
#     # Create Lightning module
#     lightning_module = SatRainEstimationModule(
#         model=efficient_net_model,
#         loss=nn.MSELoss(),
#         approach=compute_config.approach,
#     )
#     #experiment_name = lightning_module.experiment_name
#     experiment_name = f"{lightning_module.experiment_name}_efficient_net_v2_{model_size}"

#     results = []
#     domains = ["austria"]#, "conus", "korea"]
#     for domain in domains:
#         LOGGER.info("Running evaluation for domain '%s'.", domain)
#         evaluator = data_module.get_evaluator(domain)
#         evaluator.evaluate(
#             lightning_module.get_retrieval_fn(satrain_config, compute_config),
#             batch_size=compute_config.batch_size,
#             tile_size=(64, 64)
#         )
#         results.append(evaluator.get_results())

#     results = xr.concat(results, dim="domain")
#     results["domain"] = domains

#     output_path = Path(".") / "test_results"
#     output_path.mkdir(exist_ok=True)
#     results.to_netcdf(output_path / f"{experiment_name}.nc")


# if __name__ == '__main__':
#     main()
