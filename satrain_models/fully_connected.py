"""
satrain_models.fully_connected
==============================

Provides an implementation of a basic PyTorch Fully Connected Network.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# Try to import TOML library (tomllib in Python 3.11+, tomli for older versions)
try:
    import tomllib
    TOML_AVAILABLE = True
except ImportError:
    try:
        import tomli as tomllib
        TOML_AVAILABLE = True
    except ImportError:
        TOML_AVAILABLE = False
        tomllib = None


@dataclass
class FullyConnectedConfig:
    """Configuration for FullyConnected model."""
    hidden_dims: List[int] = None
    dropout: float = 0.0

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [16, 8, 4]

    def __str__(self) -> str:
        """Generate a meaningful string representation for model naming."""
        dims_str = "x".join(map(str, self.hidden_dims))
        dropout_str = f"drop{self.dropout}" if self.dropout > 0 else "nodrop"
        return f"{dims_str}_{dropout_str}"

    @property
    def model_name(self) -> str:
        """Alias for string representation - used for model naming."""
        return str(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FullyConnectedConfig":
        """Create FullyConnectedConfig from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_toml_file(cls, toml_path: Union[str, Path]) -> "FullyConnectedConfig":
        """Create FullyConnectedConfig from TOML file."""
        if not TOML_AVAILABLE:
            raise ImportError(
                "TOML library not available. Install with: pip install tomli"
            )

        toml_path = Path(toml_path)
        if not toml_path.exists():
            raise FileNotFoundError(f"TOML file not found: {toml_path}")

        with open(toml_path, "rb") as f:
            config_dict = tomllib.load(f)

        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert FullyConnectedConfig to dictionary representation."""
        return asdict(self)


class FullyConnectedBlock(nn.Module):
    """
    A fully connected block consisting of a Linear -> BatchNorm -> ReLU sequence.
    """

    def __init__(self, in_features, out_features, dropout=0.0):
        super().__init__()
        layers = [
            nn.Linear(in_features, out_features, bias=False),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class FullyConnectedNetwork(nn.Module):
    """
    A basic fully connected feed-forward network.
    """

    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0):
        """
        Args:
            input_dim (int): Number of input features.
            hidden_dims (list[int]): List of hidden layer sizes.
            output_dim (int): Number of output features.
            dropout (float): Dropout rate applied after each hidden layer. Default: 0.0
        """
        super(FullyConnectedNetwork, self).__init__()

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(FullyConnectedBlock(in_dim, h_dim, dropout=dropout))
            in_dim = h_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, output_dim)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        x = self.feature_extractor(x)
        x = self.output_layer(x)
        #x = x.squeeze()
        return x


def create_fully_connected(input_dim=3, hidden_dims=[16, 8, 4], output_dim=1, dropout=0.0):
    """Create a Fully Connected Network model.

    Args:
        input_dim (int): Number of input features. Default: 3
        hidden_dims (list[int]): List of hidden layer sizes. Default: [16, 8, 4]
        output_dim (int): Number of output features. Default: 1
        dropout (float): Dropout rate. Default: 0.0

    Returns:
        FullyConnectedNetwork: FCN model instance
    """
    return FullyConnectedNetwork(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout=dropout,
    )
