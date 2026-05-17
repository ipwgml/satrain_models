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

# Upstream bug: satrain.evaluation.Evaluator.get_input_files checks
# `hasattr(self, "geo_ir_t_gridded")` for the geo_ir slot (lines 981-982 in
# satrain 2025.x). For geo_ir-only configs the `_t` attribute is never set,
# so the geo_ir file path is silently None, and test scenes come back with
# no obs_geo_ir variable. We locally override Evaluator.get_input_files
# below with corrected attribute keys until a fix lands upstream in
# ipwgml/satrain.
import satrain.evaluation as _se

def _fixed_get_input_files(self, index):
    if len(self) <= index:
        raise IndexError("'index' exceeds number of available collocation scenes.")
    return _se.InputFiles(
        self.target_gridded[index],
        self.target_on_swath[index] if hasattr(self, "target_on_swath") else None,
        self.gmi_gridded[index] if hasattr(self, "gmi_gridded") else None,
        self.gmi_on_swath[index] if hasattr(self, "gmi_on_swath") else None,
        self.atms_gridded[index] if hasattr(self, "atms_gridded") else None,
        self.atms_on_swath[index] if hasattr(self, "atms_on_swath") else None,
        self.ancillary_gridded[index] if hasattr(self, "ancillary_gridded") else None,
        self.ancillary_on_swath[index] if hasattr(self, "ancillary_on_swath") else None,
        self.geo_gridded[index] if hasattr(self, "geo_gridded") else None,
        self.geo_on_swath[index] if hasattr(self, "geo_on_swath") else None,
        self.geo_t_gridded[index] if hasattr(self, "geo_t_gridded") else None,
        self.geo_t_on_swath[index] if hasattr(self, "geo_t_on_swath") else None,
        self.geo_ir_gridded[index] if hasattr(self, "geo_ir_gridded") else None,
        self.geo_ir_on_swath[index] if hasattr(self, "geo_ir_on_swath") else None,
        self.geo_ir_t_gridded[index] if hasattr(self, "geo_ir_t_gridded") else None,
        self.geo_ir_t_on_swath[index] if hasattr(self, "geo_ir_t_on_swath") else None,
    )

_se.Evaluator.get_input_files = _fixed_get_input_files

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

    # weights_only=False is required for self-sufficient .ckpt files: they embed
    # the satrain_config dict (saved via save_hyperparameters), which contains
    # numpy arrays / custom types that torch's weights-only allowlist rejects.
    # These checkpoints are local training artifacts, so trusting the pickle is
    # safe.
    loaded = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    state = loaded["state_dict"]

    # Remove model prefix for checkpoint files.
    if model_path.suffix == ".ckpt":
        state = {key[6:]: val for key, val in state.items()}

    # satrain_config lives at the top level for hand-saved .pt files, and
    # under hyper_parameters for self-sufficient .ckpt files saved via
    # save_hyperparameters.
    if "satrain_config" in loaded:
        sc_dict = loaded["satrain_config"]
    elif "hyper_parameters" in loaded and "satrain_config" in loaded["hyper_parameters"] and loaded["hyper_parameters"]["satrain_config"] is not None:
        sc_dict = loaded["hyper_parameters"]["satrain_config"]
    else:
        raise KeyError(
            f"Could not find 'satrain_config' in {model_path}. "
            "Re-save the model with the updated train.py."
        )

    # The geo_ir checkpoint saved its retrieval_input name as 'geoir' because
    # SatRainConfig.to_dict() used class.__name__.lower() (which drops the
    # underscore in 'GeoIR'). satrain.parse_retrieval_inputs rejects 'geoir',
    # so rewrite it here so the existing ckpt loads without re-training.
    for _r in sc_dict.get("retrieval_input", []) or []:
        if isinstance(_r, dict) and _r.get("name") == "geoir":
            _r["name"] = "geo_ir"

    satrain_config = SatRainConfig(**sc_dict)
    # NOTE 2026-05-14: removed an unconditional
    # `satrain_config.retrieval_input[0].include_angles = False` override that
    # was here previously. It was a no-op for geo_ir (GeoIR doesn't read the
    # attribute) but broke atms testing by halving the feature count from 18
    # (9 BTs + 9 angles) to 9, producing a conv_in shape mismatch on load.
    # The saved ckpt config is the source of truth; don't mutate it on read.
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

    # Create Lightning module. Pass through the saved experiment 'name' so the
    # downstream output filename actually identifies the model under test —
    # otherwise experiment_name falls back to a generic short form like
    # 'adamw_warmup_cosine_annealing_restarts_v00' and consecutive test runs
    # on different checkpoints clobber each other's .nc results.
    saved_name = loaded.get("hyper_parameters", {}).get("name")
    lightning_module = SatRainEstimationModule(
        model=resnet_model,
        loss=nn.MSELoss(),
        approach=compute_config.approach,
        name=saved_name,
    )
    # Result filename: <saved_experiment>_<ckpt_stem>.nc — identifies both
    # the training run and the specific checkpoint within it.
    experiment_name = f"{lightning_module.experiment_name}_{model_path.stem}"

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
