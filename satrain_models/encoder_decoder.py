#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Protocol, Union, Optional, Tuple, Dict, Any

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


class BlockFactory(Protocol):
    """
    The block factory creates the convolutional block in an encoder-decoder type neural network.

    The block factory creates a block with a given number of input channels, and output channes, and
    perform optional downsampling of the inputs.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            downsample: Optional[Tuple[int, int]] = None,
            stage_ind: int = 0,
            block_ind: int = 0,
    ) -> None:
        """
        Initialize a convolution block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            downsample: Optional tuple (height_factor, width_factor) for downsampling
            stage_ind: Zero-based index identifying the current stage.
            block_ind: Index of this block within the current stage (0-indexed)
        """


class Conv2dBnReLU(nn.Module):
    """
    Convolution block consisting of two Conv2d-BatchNorm-ReLU sequences.
    Optionally performs downsampling via MaxPool2d or strided convolution.

    This block is used in the original U-Net.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            downsample: Optional[Tuple[int, int]] = None,
            stage_ind: int = 0,
            block_ind: int = 0
    ):
        super().__init__()
        
        mid_channels = out_channels
        
        # First convolution
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second convolution
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Optional downsampling
        self.downsample = None
        if downsample is not None:
            self.downsample = nn.MaxPool2d(kernel_size=downsample, stride=downsample)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.downsample is not None:
            x = self.downsample(x)

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))

        return x


class ResNeXtBlock(nn.Module):
    """
    ResNeXt block with grouped convolutions and residual connection.
    
    Architecture: 1x1 conv -> 3x3 grouped conv -> 1x1 conv + residual connection
    Optionally performs downsampling via strided convolution in the residual path.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: Optional[Tuple[int, int]] = None,
        stage_ind: int = 0,
        block_ind: int = 0,
        cardinality: int = 32,
        bottleneck_width: int = 4,
    ):
        super().__init__()
        
        # Calculate bottleneck channels
        width = int(out_channels * (bottleneck_width / 64.0)) * cardinality
        
        # Determine stride for downsampling
        stride = downsample if downsample is not None else (1, 1)
        
        # Main path: 1x1 -> 3x3 grouped -> 1x1
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        
        self.conv2 = nn.Conv2d(
            width, width, 
            kernel_size=3, stride=stride, padding=1, 
            groups=cardinality, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width)
        
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Residual connection
        self.residual = None
        if in_channels != out_channels or downsample is not None:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, 
                    kernel_size=3, stride=stride, bias=False, padding=1
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        # Main path
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # Residual path
        if self.residual is not None:
            residual = self.residual(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class Conv2dLnGELU(nn.Module):
    """
    Convolution block consisting of two Conv2d-LayerNorm-GELU sequences.
    Optionally performs downsampling via MaxPool2d or strided convolution.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            downsample: Optional[Tuple[int, int]] = None,
            stage_ind: int = 0,
            block_ind: int = 0
    ):
        super().__init__()
        
        mid_channels = out_channels
        
        # First convolution
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True)
        # LayerNorm expects (N, C, H, W) format, we'll use LayerNorm with normalized_shape=[C, H, W]
        # but since H, W vary, we'll normalize over channel dimension only
        self.ln1 = nn.GroupNorm(1, mid_channels)  # GroupNorm with 1 group = LayerNorm over channels
        self.gelu1 = nn.GELU()
        
        # Second convolution
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.ln2 = nn.GroupNorm(1, out_channels)
        self.gelu2 = nn.GELU()
        
        # Optional downsampling
        self.downsample = None
        if downsample is not None:
            h_factor, w_factor = downsample
            if h_factor == w_factor == 2:
                # Use MaxPool2d for 2x2 downsampling
                self.downsample = nn.MaxPool2d(2)
            else:
                # Use strided convolution for other downsampling factors
                self.downsample = nn.Conv2d(
                    out_channels, out_channels, 
                    kernel_size=3, stride=(h_factor, w_factor), padding=1, bias=True
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.downsample is not None:
            x = self.downsample(x)

        x = self.gelu1(self.ln1(self.conv1(x)))
        x = self.gelu2(self.ln2(self.conv2(x)))
        
        return x


class Conv2dLnReLU(nn.Module):
    """
    Convolution block consisting of two Conv2d-LayerNorm-ReLU sequences.
    Optionally performs downsampling via MaxPool2d or strided convolution.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            downsample: Optional[Tuple[int, int]] = None,
            stage_ind: int = 0,
            block_ind: int = 0
    ):
        super().__init__()
        
        mid_channels = out_channels
        
        # First convolution
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True)
        self.ln1 = nn.GroupNorm(1, mid_channels)  # GroupNorm with 1 group = LayerNorm over channels
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second convolution
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True)
        self.ln2 = nn.GroupNorm(1, out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Optional downsampling
        self.downsample = None
        if downsample is not None:
            h_factor, w_factor = downsample
            if h_factor == w_factor == 2:
                # Use MaxPool2d for 2x2 downsampling
                self.downsample = nn.MaxPool2d(2)
            else:
                # Use strided convolution for other downsampling factors
                self.downsample = nn.Conv2d(
                    out_channels, out_channels, 
                    kernel_size=3, stride=(h_factor, w_factor), padding=1, bias=True
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.downsample is not None:
            x = self.downsample(x)

        x = self.relu1(self.ln1(self.conv1(x)))
        x = self.relu2(self.ln2(self.conv2(x)))

        return x


class EncoderStage(nn.Module):
    """
    An encoder stage containing multiple convolution blocks.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            block_factory: Callable[[int, int], nn.Module],
            depth: int,
            downsample: Optional[Tuple[int, int]] = None,
            stage_ind: int = 0,
    ):
        super().__init__()
        blocks = []
        current_channels = in_channels
        for block_ind in range(depth):
            block = block_factory(
                in_channels=current_channels,
                out_channels=out_channels,
                downsample=downsample,
                stage_ind=stage_ind,
                block_ind=block_ind
            )
            blocks.append(block)
            current_channels = out_channels
            downsample = None
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward tensor through stage.
        """
        return self.blocks.forward(x)


class DecoderStage(nn.Module):
    """
    A decoder stage containing several convolution blocks and an upsampling block.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            sc_channels: int,
            block_factory: Callable[[int, int], nn.Module],
            depth: int,
            upsample: Optional[Tuple[int, int]] = None,
            stage_ind: int = 0,
    ):
        """
        Args:
            stage_ind: The index of the stage.
            in_channels: The number of incoming channels from the previous stage.
            out_channels: The outgoing number of channels in the stage.
            sc_channels: The channels from the shortcut.
            block_factor: Convblock class to use to create the convolutional blocks.
            depth: The number of blocks in the stage.
            upsamle: Tuple defining the upsampling to apply in the stage.
        """
        super().__init__()
        blocks = []
        current_channels = in_channels + sc_channels
        for block_ind in range(depth):
            block = block_factory(
                in_channels=current_channels,
                out_channels=out_channels,
                downsample=None,
                stage_ind=stage_ind,
                block_ind=block_ind
            )
            blocks.append(block)
            current_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

        self.upsample = None
        if upsample is not None:
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x: torch.Tensor, x_sc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: The output from the previous stage.
            x_sc: The shortcut tensor from the corresponding encoder stage.

        Return:
            The output from the stage.
        """
        if self.upsample is not None:
            x = self.upsample(x)
        x = torch.cat((x, x_sc), 1)
        return self.blocks(x)


class EncoderDecoder(nn.Module):
    """
    The EncoderDecoder class represents a generic UNet model that can be configured to use different
    convolutional blocks.

    A specific encoder-decoder model is specified by:

     1. a BlockFactory that is used to create the convolution blocks used in the model
     2. the number of input channels
     3. a sequence of channels specifying the number of channels at the end of each encoder stage
     4. a sequence of depths specifying the depth of each stage
     5. the number of output channels

    Downsampling is applied by the first convolution block of each stage except
    for the first stage. Upsampling is performed using bilinear interpolation
    but can be configured to use transposed convolutions.
    """

    def __init__(
        self,
        block_factory: type[BlockFactory],
        in_channels: int,
        channels: List[int],
        depths: List[int],
        out_channels: int = 1,
        bilinear: bool = True,
    ):
        """
        Initialize the EncoderDecoder model.
        
        Args:
            block_factory: Class for creating convolution blocks
            in_channels: Number of input channels
            channels: List of channels for each encoder stage
            depths: List of depths for each encoder stage
            out_channels: Number of output channels
            bilinear: Whether to use bilinear upsampling (vs transposed convolution)
        """
        super().__init__()
        
        if len(channels) != len(depths):
            raise ValueError("channels and depths must have the same length")
        
        self.bilinear = bilinear
        self.out_channels = out_channels
        
        # Build encoder stages
        self.encoder_stages = nn.ModuleList()
        current_channels = in_channels
        
        for stage_idx, (stage_channels, stage_depth) in enumerate(zip(channels, depths)):
            stage = EncoderStage(
                block_factory=block_factory,
                in_channels=current_channels,
                out_channels=stage_channels,
                depth=stage_depth,
                downsample=(2, 2) if 0 < stage_idx else None,
                stage_ind=stage_idx,
            )
            self.encoder_stages.append(stage)
            current_channels = stage_channels
        
        # Build decoder stages (reverse order, skip last encoder stage as it's the bottleneck)
        self.decoder_stages = nn.ModuleList()
        decoder_channels = list(reversed(channels[:-1]))
        decoder_depths = list(reversed(depths[:-1]))
        
        for stage_idx, (stage_channels, stage_depth) in enumerate(zip(decoder_channels, decoder_depths)):
            stage = DecoderStage(
                in_channels=current_channels,
                out_channels=stage_channels,
                sc_channels=stage_channels,
                block_factory=block_factory,
                depth=stage_depth,
                upsample=(2, 2),
                stage_ind=len(channels) - 1 - stage_idx,
            )
            self.decoder_stages.append(stage)
            current_channels = stage_channels
        self.output_conv = nn.Conv2d(current_channels, out_channels, kernel_size=1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder-decoder network.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        skip_connections = []
        current = x
        for stage in self.encoder_stages:
            current = stage(current)
            skip_connections.append(current)
        
        for stage, skip in zip(self.decoder_stages, reversed(skip_connections[:-1])):
            current = stage(current, skip)

        return self.output_conv(current)

    @property
    def num_parameters(self) -> int:
        """Return the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())
    
    @property 
    def num_trainable_parameters(self) -> int:
        """Return the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@dataclass
class EncoderDecoderConfig:
    """Configuration for EncoderDecoder model."""
    
    block_factory: str
    in_channels: int
    channels: List[int]
    depths: List[int]
    out_channels: int = 1
    bilinear: bool = True
    
    def __str__(self) -> str:
        """Generate a meaningful string representation for model naming."""
        # Extract block type (remove 'Block' suffix if present)
        block_name = self.block_factory.replace('Block', '').lower()
        
        # Create channel signature: input -> [stages] -> output
        stages_str = 'x'.join(map(str, self.channels))
        
        # Create depth signature
        if all(d == self.depths[0] for d in self.depths):
            # All depths are the same
            depth_str = f"d{self.depths[0]}"
        else:
            # Different depths per stage
            depth_str = 'd' + 'x'.join(map(str, self.depths))
        
        # Upsampling method
        upsample = "bilinear" if self.bilinear else "transpose"
        
        # Combine all parts
        return f"{block_name}_in{self.in_channels}_ch{stages_str}_{depth_str}_out{self.out_channels}_{upsample}"
    
    @property
    def model_name(self) -> str:
        """Alias for string representation - used for model naming."""
        return str(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EncoderDecoderConfig":
        """Create EncoderDecoderConfig from dictionary.
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            EncoderDecoderConfig instance
        """
        return cls(**config_dict)
    
    @classmethod
    def from_toml_file(cls, toml_path: Union[str, Path]) -> "EncoderDecoderConfig":
        """Create EncoderDecoderConfig from TOML file.
        
        Args:
            toml_path: Path to the TOML configuration file
            
        Returns:
            EncoderDecoderConfig instance
            
        Raises:
            ImportError: If TOML library is not available
            FileNotFoundError: If the TOML file doesn't exist
        """
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
    
    @classmethod
    def from_toml_string(cls, toml_string: str) -> "EncoderDecoderConfig":
        """Create EncoderDecoderConfig from TOML string.
        
        Args:
            toml_string: TOML configuration as string
            
        Returns:
            EncoderDecoderConfig instance
            
        Raises:
            ImportError: If TOML library is not available
        """
        if not TOML_AVAILABLE:
            raise ImportError(
                "TOML library not available. Install with: pip install tomli"
            )
        
        config_dict = tomllib.loads(toml_string)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert EncoderDecoderConfig to dictionary representation.
        
        Returns:
            Dictionary representation of the configuration
        """
        return asdict(self)
    
    def to_toml_file(self, toml_path: Union[str, Path]) -> None:
        """Write EncoderDecoderConfig to TOML file.
        
        Args:
            toml_path: Path where to write the TOML configuration file
            
        Raises:
            ImportError: If TOML writing library is not available
        """
        try:
            import tomli_w
        except ImportError:
            raise ImportError(
                "TOML writing library not available. Install with: pip install tomli-w"
            )
        
        toml_path = Path(toml_path)
        config_dict = self.to_dict()
        
        with open(toml_path, "wb") as f:
            tomli_w.dump(config_dict, f)
