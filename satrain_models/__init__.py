"""SatRain Models - ML model implementations for the SatRain dataset."""

from .config import ComputeConfig, SatRainConfig
from .datamodule import SatRainDataModule
from .lightning import SatRainEstimationModule
from .swinunet import SwinUnet, create_swinunet
from .tensorboard_to_netcdf import (
    extract_all_training_runs,
    extract_scalars_from_tensorboard,
    scalars_to_netcdf,
    tensorboard_to_netcdf,
)
from .unet import UNet, create_unet

__version__ = "0.1.0"
__author__ = "SatRain Contributors"

__all__ = [
    "UNet",
    "create_unet",
    "SwinUnet",
    "create_swinunet",
    "SatRainConfig",
    "SatRainEstimationModule",
    "SatRainDataModule",
    "extract_scalars_from_tensorboard",
    "scalars_to_netcdf",
    "tensorboard_to_netcdf",
    "extract_all_training_runs",
]
