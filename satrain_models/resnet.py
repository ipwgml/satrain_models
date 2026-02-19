"""ResNet implementation for precipitation prediction.

This module provides a ResNet architecture adapted for precipitation estimation
from satellite data. The architecture preserves spatial dimensions throughout
to maintain the on-swath geometry of the input data (417 x 221 pixels).
This, however, can be overwritten in the train.py into something else (e.g., 64x64)

Created: January 2026
questions: veljko.petkovic@umd.edu
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ResNetConfig:
    """Configuration for ResNet model."""
    
    n_channels: int
    n_outputs: int = 1
    blocks_per_layer: int = 3
    n_layers: int = 4
    base_channels: int = 64

    def __str__(self) -> str:
        """Generate a meaningful string representation for model naming."""
        return f"resnet_{self.n_layers}layer_{self.blocks_per_layer}blocks_ch{self.base_channels}_in{self.n_channels}_out{self.n_outputs}"

    @property
    def model_name(self) -> str:
        """Alias for string representation - used for model naming."""
        return str(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ResNetConfig":
        """Create ResNetConfig from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            ResNetConfig instance
        """
        return cls(**config_dict)

    @classmethod
    def from_toml_file(cls, toml_path: Union[str, Path]) -> "ResNetConfig":
        """Create ResNetConfig from TOML file.
        
        Args:
            toml_path: Path to the TOML configuration file
            
        Returns:
            ResNetConfig instance
            
        Raises:
            ImportError: If TOML library is not available
            FileNotFoundError: If the TOML file doesn't exist
        """
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib
            except ImportError:
                raise ImportError("No TOML library available. Please install tomli.")

        toml_path = Path(toml_path)
        if not toml_path.exists():
            raise FileNotFoundError(f"Config file not found: {toml_path}")

        with open(toml_path, "rb") as f:
            config_dict = tomllib.load(f)

        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert ResNetConfig to dictionary.
        
        Returns:
            Dictionary containing configuration parameters
        """
        return {
            "n_channels": self.n_channels,
            "n_outputs": self.n_outputs,
            "blocks_per_layer": self.blocks_per_layer,
            "n_layers": self.n_layers,
            "base_channels": self.base_channels,
        }


class ResidualBlock(nn.Module):
    """Basic residual block with skip connection.

    A residual block consists of two convolutions with batch normalization
    and a skip connection that allows the gradient to flow directly through.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        stride (int): Stride for the first convolution (default: 1)
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Use reflection padding instead of zero padding to reduce edge artifacts
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            padding=0, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Skip connection: identity if dimensions match, else adapt via 1x1 conv
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """Forward pass with residual connection.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W)
        """
        identity = self.skip(x)
        out = self.pad1(x)
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.pad2(out)
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet encoder for precipitation prediction.

    This ResNet architecture is adapted for satellite precipitation estimation.
    It maintains spatial dimensions throughout to preserve the on-swath geometry
    of the input data. The architecture uses residual blocks arranged in layers
    to extract hierarchical features.

    Args:
        n_channels (int): Number of input channels (e.g., 13 for satellite bands)
        n_outputs (int): Number of output channels (e.g., 1 for precipitation)
        blocks_per_layer (int): Number of residual blocks per layer (default: 2)
        n_layers (int): Number of ResNet layers (3 or 4, default: 4)
        base_channels (int): Number of channels in first layer (default: 64)
    """

    def __init__(self, n_channels, n_outputs, blocks_per_layer=2, n_layers=4, base_channels=64):
        super().__init__()
        self.n_channels = n_channels
        self.n_outputs = n_outputs
        self.n_layers = n_layers

        # Initial convolution: map from input channels to base_channels feature maps
        # Preserve spatial dimensions with reflection padding 
        self.conv_in = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_channels, base_channels, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # ResNet layers 
        # No downsampling (stride=1); preserves spatial dimensions
        c = base_channels  # 64
        self.layer1 = self._make_layer(c, c, blocks_per_layer, stride=1)
        self.layer2 = self._make_layer(c, c * 2, blocks_per_layer, stride=1)  # 128
        self.layer3 = self._make_layer(c * 2, c * 4, blocks_per_layer, stride=1)  # 256

        # 4th layer (512 channels)
        if n_layers >= 4:
            self.layer4 = self._make_layer(c * 4, c * 8, blocks_per_layer, stride=1)  # 512
            final_channels = c * 8  # 512
        else:
            self.layer4 = None
            final_channels = c * 4  # 256

        # Output convolution: map from final feature maps to output channels
        # Use reflection padding 
        self.conv_out = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(final_channels, c * 2, kernel_size=3, padding=0),  # 128
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, n_outputs, kernel_size=1)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """Create a sequence of residual blocks.

        Args:
            in_channels (int): Number of input channels for the layer
            out_channels (int): Number of output channels for the layer
            blocks (int): Number of residual blocks in this layer
            stride (int): Stride for the first block (default: 1)

        Returns:
            nn.Sequential: Sequential container of residual blocks
        """
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the ResNet.

        Args:
            x (torch.Tensor): Input tensor of shape (B, n_channels, H, W)
                where B is batch size, H=417, W=221 for SatRain data

        Returns:
            torch.Tensor: Output tensor of shape (B, n_outputs, H, W)
                Maintains same spatial dimensions as input
        """
        x = self.conv_in(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer4 is not None:
            x = self.layer4(x)
        x = self.conv_out(x)
        return x


def create_resnet(n_channels=3, n_outputs=1, blocks_per_layer=3, n_layers=4, base_channels=64):
    """Factory function to create a ResNet model.

    Args:
        n_channels (int): Number of input channels. Default: 3
        n_outputs (int): Number of output channels. Default: 1
        blocks_per_layer (int): Number of blocks per layer. Default: 3
        n_layers (int): Number of ResNet layers (3 or 4). Default: 4
        base_channels (int): Number of channels in first layer. Default: 64

    Returns:
        ResNet: ResNet model instance ready for training or inference

    Example:
        >>> model = create_resnet(n_channels=13, n_outputs=1)
        >>> x = torch.randn(2, 13, 64, 64)
        >>> y = model(x)
        >>> assert y.shape == (2, 1, 64, 64)

    """
    return ResNet(
        n_channels=n_channels,
        n_outputs=n_outputs,
        blocks_per_layer=blocks_per_layer,
        n_layers=n_layers,
        base_channels=base_channels,
    )


def create_resnet_from_config(model_config: ResNetConfig, num_features: int) -> ResNet:
    """Create ResNet model from configuration.
    
    Args:
        model_config: ResNet configuration object
        num_features: Number of input features (overrides config n_channels)
        
    Returns:
        ResNet: Configured ResNet model
    """
    return ResNet(
        n_channels=num_features,
        n_outputs=model_config.n_outputs,
        blocks_per_layer=model_config.blocks_per_layer,
        n_layers=model_config.n_layers,
        base_channels=model_config.base_channels,
    )
