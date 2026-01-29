"""
satrain_models.unetpp
=====================

Provides a generic implementation of the UNet++ architecture. The UNet++
 architecture is a generalization of the UNet architecture that densely
connects all stages.

Instead of a single compute branch that sequentially down- and upsamples
the input tensor, the UNet++ architecture processes tensors at all scales
in parallel, with all multi-scale outputs from the previous stage forming
the input for the subsequent stage.

    B - B - B - B - B
      - B x B x B -
          - B -

"""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import torch
from torch import nn

from .unet import (
    DoubleConv,
    Down,
    Up,
    OutConv,
)
from .encoder_decoder import (
    BlockFactory,
    EncoderStage,
    DecoderStage,
    EncoderDecoderConfig
)


def scale_to_str(scale: float) -> str:
    """
    Helper function to format a floating point value as a suitable key for a
    torch.ModuleDict.

    Args:
        scale: A floating point number defining the scale.

    Return:
        A string with the decimal point replace by a comma.
    """
    return str(scale).replace(".", ",")


class DenseEncoderDecoder(nn.Module):
    """
    The DenseEncoderDecoder class represents a generic UNet++ architecture
    that can be configured to use different convolutional blocks.

    A specific encoder-decoder model is specified by:

     1. a BlockFactory that is used to create the convolution blocks used in the model
     2. the number of input channels
     3. A sequence of channels specifying the number of channels at each scale.
     4. A sequence of depths specifying the number of blocks forming a stage
        at each scale.
     5. The number of output channels

    Downsampling is performed using average pooling. Upsampling is performed using
    bilinear interpolation.
    """

    def __init__(
        self,
        block_factory: type[BlockFactory],
        in_channels: int,
        channels: List[int],
        depths: List[int],
        out_channels: int = 1,
        bilinear: bool = None,
    ):
        """
        Initialize the EncoderDecoder model.

        Args:
            block_factory: Class for creating convolution blocks
            in_channels: Number of input channels
            channels: List of channels for each encoder stage
            depths: List of depths for each encoder stage
            out_channels: Number of output channels
            bilinear: Not used
        """
        super().__init__()

        if len(channels) != len(depths):
            raise ValueError("channels and depths must have the same length")

        self.out_channels = out_channels
        n_stages = len(channels)

        # Build encoder stages
        self.encoder_stages = nn.ModuleList()
        for stage_idx, (chans_stage, stage_depth) in enumerate(
            zip(channels, depths)
        ):
            stages = nn.ModuleList()
            if stage_idx == 0:
                chans_in = in_channels
            else:
                chans_in = sum(channels[:stage_idx])

            for level_idx in range(stage_idx + 1):
                chans_stage = channels[level_idx]
                stage = EncoderStage(
                    block_factory=block_factory,
                    in_channels=chans_in,
                    out_channels=chans_stage,
                    depth=stage_depth,
                    downsample=None,
                    stage_ind=stage_idx,
                )
                stages.append(stage)
            self.encoder_stages.append(stages)

        # Build decoder stages
        self.decoder_stages = nn.ModuleList()
        for stage_idx in range(len(channels) - 1):
            chans_in = sum(channels[:n_stages - stage_idx])
            stages = nn.ModuleList()
            for lvl_idx in range(n_stages - 1 - stage_idx):
                chans_lvl = channels[lvl_idx]
                depth_lvl = depths[lvl_idx]
                stage = DecoderStage(
                    in_channels=chans_in,
                    out_channels=chans_lvl,
                    sc_channels=0,
                    block_factory=block_factory,
                    depth=depth_lvl,
                    stage_ind=len(channels) - 1 - stage_idx,
                )
                stages.append(stage)
            self.decoder_stages.append(stages)

        self.output_conv = nn.Conv2d(
            channels[0],
            out_channels,
            kernel_size=1
        )

        self.scalers = nn.ModuleDict()
        for in_scale in [2 ** ind for ind in range(n_stages)]:
            for out_scale in [2 ** ind for ind in range(n_stages)]:
                scale_diff = in_scale / out_scale
                if 1.0 < scale_diff:
                    self.scalers[scale_to_str(scale_diff)] = nn.Upsample(
                        scale_factor=scale_diff,
                        mode="bilinear"
                    )
                elif scale_diff < 1.0:
                    self.scalers[scale_to_str(scale_diff)] = nn.AvgPool2d(
                        kernel_size=int(1.0 / scale_diff),
                        stride=int(1.0 / scale_diff),
                    )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder-decoder network.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        skip_connections = []
        current = [x]
        for stages in self.encoder_stages + self.decoder_stages:
            results = []
            for stage_ind, stage in enumerate(stages):
                stage_scale = 2 ** stage_ind
                x_scld = []
                for tnsr_ind, tnsr in enumerate(current):
                    in_scale = 2 ** tnsr_ind
                    scale_diff = stage_scale / in_scale
                    if (1 < scale_diff) or (scale_diff < 1):
                        x_scld.append(self.scalers[scale_to_str(scale_diff)](tnsr))
                    else:
                        x_scld.append(tnsr)
                x_scld = torch.cat(x_scld, dim=1)
                results.append(stage(x_scld))
            current = results
        return self.output_conv(current[0])

    @property
    def num_parameters(self) -> int:
        """Return the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Return the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
