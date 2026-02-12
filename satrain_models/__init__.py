"""SatRain Models - ML model implementations for the SatRain dataset."""

from .config import ComputeConfig, SatRainConfig
from .datamodule import SatRainDataModule
from .lightning import SatRainEstimationModule
from .random_forest import RandomForestRetrieval, create_random_forest_retrieval
from .tensorboard_to_netcdf import (
    extract_all_training_runs,
    extract_scalars_from_tensorboard,
    scalars_to_netcdf,
    tensorboard_to_netcdf,
)
from .unet import UNet, create_unet
from .fully_connected import FullyConnectedNetwork, create_fully_connected
from .xgboost import XGBoostRetrieval, create_xgboost
from .efficient_net_v2 import create_efficient_net_v2 
from .convnext import create_convnext

__version__ = "0.1.0"
__author__ = "SatRain Contributors"

__all__ = [
    "UNet",
    "create_unet",
    "FullyConnectedNetwork",
    "create_fully_connected",
    "XGBoostRetrieval",
    "create_xgboost",
    "RandomForestRetrieval",
    "create_random_forest_retrieval",
    "create_swinunet",
    "SatRainConfig",
    "SatRainEstimationModule",
    "SatRainDataModule",
    "extract_scalars_from_tensorboard",
    "scalars_to_netcdf",
    "tensorboard_to_netcdf",
    "extract_all_training_runs",
    "create_efficient_net_v2",
    "create_convnext" 
]
