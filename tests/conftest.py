import os
from pathlib import Path

import pytest

import satrain
from satrain.data import (
    enable_testing,
)

enable_testing()

@pytest.fixture(scope="session")
def satrain_test_data():
    """
    Creates a temporary directory to which SatRain files are downloaded.
    """
    setattr(satrain.config, "CONFIG_DIR", Path("."))
