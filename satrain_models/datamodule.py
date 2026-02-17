"""
satrain_models.datamodule
=========================

Lightning DataModule for SatRain dataset with automatic configuration from TOML files.
"""

from pathlib import Path
from typing import Optional, Union

import lightning as L
import numpy as np
from satrain.data import load_tabular_data
from satrain.evaluation import Evaluator
from satrain.pytorch.datasets import SatRainSpatial, SatRainTabular
from torch.utils.data import DataLoader

from .config import SatRainConfig


class SatRainDataModule(L.LightningDataModule):
    """
    Lightning DataModule for SatRain dataset.

    This data module handles the creation of train, validation, and test datasets
    from a SatRainConfig configuration object or TOML file.
    """

    def __init__(
        self,
        config: Union[SatRainConfig, str, Path],
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ):
        """
        Initialize the SatRain DataModule.

        Args:
            config: SatRainConfig object or path to TOML configuration file
            batch_size: Batch size for data loaders
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for faster GPU transfer
            persistent_workers: Whether to keep workers alive between epochs
        """
        super().__init__()

        # Load config if path provided
        if isinstance(config, SatRainConfig):
            self.config = config
        elif isinstance(config, (str, Path)):
            self.config = SatRainConfig.from_toml_file(config)
        elif isinstance(config, dict):
            self.config = SatRainConfig.from_dict(config)
        else:
            raise ValueError(
                "'config' argument should either be a SatRainConfig object, "
                " a path pointing to a .toml file or a dict."
            )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and (num_workers > 0)

        # Datasets will be created in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _get_retrieval_input(self) -> list:
        """Extract retrieval input configuration from config."""
        return self.config.retrieval_input

    def _create_dataset(self, split: str, augment: bool = False):
        """Create a dataset for the given split."""
        retrieval_input = self._get_retrieval_input()

        # Common dataset arguments
        dataset_kwargs = {
            "base_sensor": getattr(self.config, "base_sensor", "gmi"),
            "geometry": getattr(self.config, "geometry", "gridded"),
            "split": split,
            "subset": getattr(self.config, "subset", "xl"),
            "retrieval_input": retrieval_input,
            "target_config": getattr(self.config, "target_config", None),
            "stack": True,
            "download": True,
        }

        if self.config.format == "spatial":
            dataset_kwargs["augment"] = augment
            return SatRainSpatial(**dataset_kwargs)
        else:
            # SatRainTabular specific arguments
            dataset_kwargs["shuffle"] = augment
            dataset_kwargs["batch_size"] = self.batch_size
            return SatRainTabular(**dataset_kwargs)

    def prepare_data(self):
        """Download data if needed (called only from main process)."""
        # This method is called only from the main process
        # Data downloading is handled by the dataset classes
        pass

    def get_evaluator(self, domain: str) -> Evaluator:
        """
        Load SatRain evaluator with the configuration matching this data module.
        """
        return Evaluator(
            base_sensor=self.config.base_sensor,
            geometry=self.config.geometry,
            retrieval_input=self.config.retrieval_input,
            domain=domain,
            target_config=self.config.target_config,
        )

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for different stages."""
        if stage in (None, "fit"):
            # Training dataset with augmentation
            self.train_dataset = self._create_dataset("training", augment=True)

            # Validation dataset without augmentation
            self.val_dataset = self._create_dataset("validation", augment=False)

        if stage in (None, "test"):
            # Test dataset without augmentation
            self.test_dataset = self._create_dataset("testing", augment=False)

    def train_dataloader(self):
        """Create training data loader."""
        if self.config.format == "spatial":
            batch_size = self.batch_size
        else:
            batch_size = None
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        """Create validation data loader."""
        if self.config.format == "spatial":
            batch_size = self.batch_size
        else:
            batch_size = None
        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        """Create test data loader."""
        if self.config.format == "spatial":
            batch_size = self.batch_size
        else:
            batch_size = None
        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self):
        """Create prediction data loader (same as test for now)."""
        return self.test_dataloader()

    def load_tabular_data(self, split: str):
        """
        Load data in tabular format suitable for non-PyTorch models like XGBoost.

        Args:
            split: Dataset split ('training', 'validation', or 'testing')

        Returns:
            Tuple of (input_data, target_data) where:
            - input_data: numpy array of shape (n_samples, n_features)
            - target_data: numpy array of shape (n_samples,)
        """
        retrieval_input = self._get_retrieval_input()

        # Load tabular data using satrain.data.load_tabular_data
        input_data, target = load_tabular_data(
            dataset_name="satrain",
            base_sensor=getattr(self.config, "base_sensor", "gmi"),
            geometry=getattr(self.config, "geometry", "gridded"),
            split=split,
            subset=getattr(self.config, "subset", "xl"),
            retrieval_input=retrieval_input,
            target_config=getattr(self.config, "target_config", None),
        )

        target_time = target.time
        y = target.surface_precip.data

        input_arrays = []
        for inpt in retrieval_input:
            dataset = input_data[inpt.name].transpose("samples", ...)
            inputs = inpt.load_data(dataset, target_time=target_time)
            for array in inputs.values():
                input_arrays.append(array.T)

        X = np.concatenate(input_arrays, axis=1)
        return X, y

    @property
    def num_classes(self) -> int:
        """Number of output classes (1 for regression)."""
        return 1

    @property
    def num_features(self) -> int:
        """Number of input features/channels."""
        return self.config.num_features

    def __repr__(self) -> str:
        """String representation of the data module."""
        return (
            f"{self.__class__.__name__}(\n"
            f"  geometry={getattr(self.config, 'geometry', 'gridded')}, \n"
            f"  subset={getattr(self.config, 'subset', 'xl')}, \n"
            f"  spatial={self.spatial}, \n"
            f"  batch_size={self.batch_size}, \n"
            f"  num_workers={self.num_workers}, \n"
            f"  retrieval_input={self._get_retrieval_input()}, \n"
            f"  num_features={self.num_features}\n"
            f")"
        )
