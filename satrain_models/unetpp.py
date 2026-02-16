"""
satrain_models.unetpp
=====================

Provides a generic implementation of the UNet++ architecture. The UNet++
 architecture is a generalization of the UNet architecture that densely
connects the encoder with the decoder.

Instead of a single decoder that is connected to the encoder through
skip connections, the UNet++ model iteratively refines the features extracted
by the encoder densely connecting features at the same scale.
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
        self.encoder = nn.ModuleList()
        for stage_idx, (chans_stage, stage_depth) in enumerate(
            zip(channels, depths)
        ):
            chans_in = in_channels if stage_idx == 0 else channels[stage_idx - 1]
            chans_stage = channels[stage_idx]
            stage = EncoderStage(
                block_factory=block_factory,
                in_channels=chans_in,
                out_channels=chans_stage,
                depth=stage_depth,
                downsample=(2, 2) if 0 < stage_idx else None,
                stage_ind=stage_idx,
            )
            self.encoder.append(stage)

        # Build decoders
        self.decoders = nn.ModuleList()
        for dec_ind in range(len(channels) - 1):
            decoder = nn.ModuleList()
            for lvl_idx in range(n_stages - 1 - dec_ind):

                chans_lvl = channels[lvl_idx]
                chans_dense = (dec_ind + 1) * chans_lvl
                if lvl_idx == 0:
                    chans_dense += in_channels
                chans_up = channels[lvl_idx + 1]
                chans_in = chans_dense + chans_up
                depth_lvl = depths[lvl_idx]
                stage = DecoderStage(
                    in_channels=chans_in,
                    out_channels=chans_lvl,
                    sc_channels=0,
                    block_factory=block_factory,
                    depth=depth_lvl,
                    stage_ind=len(channels) - 1 - stage_idx,
                )
                decoder.append(stage)
            self.decoders.append(decoder)

        self.output_conv = nn.Conv2d(
            channels[0],
            out_channels,
            kernel_size=1
        )
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear"
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder-decoder network.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width)
        """
        skip_connections = {0: [x]}
        for stage_ind, stage in enumerate(self.encoder):
            x = stage(x)
            skip_connections.setdefault(stage_ind, []).append(x)

        for decoder in self.decoders:
            for stage_ind, stage in enumerate(decoder):
                dense = skip_connections[stage_ind]
                up = self.upsample(skip_connections[stage_ind + 1][-1])
                y = stage(torch.cat(dense + [up], 1))
                skip_connections[stage_ind].append(y)

        return self.output_conv(skip_connections[0][-1])

    @property
    def num_parameters(self) -> int:
        """Return the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Return the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
