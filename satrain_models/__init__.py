"""SatRain Models - ML model implementations for the SatRain dataset."""

from .unet import UNet, create_unet
from .config import SatRainConfig
from .lightning import SatRainEstimationModule
from .datamodule import SatRainDataModule
from .tensorboard_to_netcdf import (
    extract_scalars_from_tensorboard,
    scalars_to_netcdf, 
    tensorboard_to_netcdf,
    extract_all_training_runs
)

__version__ = "0.1.0"
__author__ = "SatRain Contributors"

__all__ = [
    "UNet", "create_unet", "SatRainConfig", "SatRainEstimationModule", "SatRainDataModule",
    "extract_scalars_from_tensorboard", "scalars_to_netcdf", "tensorboard_to_netcdf", 
    "extract_all_training_runs"
]