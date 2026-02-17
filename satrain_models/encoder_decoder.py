"""
satrain_models.encoder_decoder
==============================

Implements a generic encoder-decoder architecture that can be used to build UNet type models with
different convolution blocks.
"""

from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from satrain_models.config import SatRainConfig

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
        block_ind: int = 0,
    ):
        super().__init__()

        mid_channels = out_channels

        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Second convolution
        self.conv2 = nn.Conv2d(
            mid_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
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
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # Residual connection
        self.residual = None
        if in_channels != out_channels:
            if downsample is not None:
                self.residual = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=stride,
                        bias=False,
                        padding=1,
                    ),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.residual = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
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
        block_ind: int = 0,
    ):
        super().__init__()

        mid_channels = out_channels

        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, padding=1, bias=True
        )
        # LayerNorm expects (N, C, H, W) format, we'll use LayerNorm with normalized_shape=[C, H, W]
        # but since H, W vary, we'll normalize over channel dimension only
        self.ln1 = nn.GroupNorm(
            1, mid_channels
        )  # GroupNorm with 1 group = LayerNorm over channels
        self.gelu1 = nn.GELU()

        # Second convolution
        self.conv2 = nn.Conv2d(
            mid_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
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
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=(h_factor, w_factor),
                    padding=1,
                    bias=True,
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
        block_ind: int = 0,
    ):
        super().__init__()

        mid_channels = out_channels

        # First convolution
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, padding=1, bias=True
        )
        self.ln1 = nn.GroupNorm(
            1, mid_channels
        )  # GroupNorm with 1 group = LayerNorm over channels
        self.relu1 = nn.ReLU(inplace=True)

        # Second convolution
        self.conv2 = nn.Conv2d(
            mid_channels, out_channels, kernel_size=3, padding=1, bias=True
        )
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
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=(h_factor, w_factor),
                    padding=1,
                    bias=True,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.downsample is not None:
            x = self.downsample(x)

        x = self.relu1(self.ln1(self.conv1(x)))
        x = self.relu2(self.ln2(self.conv2(x)))

        return x


class Reflect(nn.Module):
    """
    Pad input by reflecting the input tensor.
    """
    def __init__(self, pad: Union[int, Tuple[int]]):
        """
        Instantiates a padding layer.

        Args:
            pad: N-tuple defining the padding added to the n-last dimensions
                of the tensor. If an int, the same padding will be added to the
                two last dimensions of the tensor.
        """
        super().__init__()
        if isinstance(pad, int):
            pad = (pad,) * 2

        full_pad = []
        for n_elems in pad:
            if isinstance(n_elems, (tuple, list)):
                full_pad += [n_elems[0], n_elems[1]]
            elif isinstance(n_elems, int):
                full_pad += [n_elems, n_elems]
            else:
                raise ValueError(
                    "Expected elements of pad tuple to be tuples of integers or integers. "
                    "Got %s.", type(n_elems)
                )

        full_pad = tuple(full_pad[::-1])
        self.pad = full_pad


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add padding to tensor.

        Args:
            x: The input tensor.

        Return:
            The padded tensor.
        """
        return nn.functional.pad(x, self.pad, "reflect")

def get_projection(
        in_channels: int,
        out_channels: int,
        stride: Tuple[int],
        anti_aliasing: bool = False,
        padding_factory: Callable[[Tuple[int]], nn.Module] = Reflect
):
    """
    Get a projection module that adapts an input tensor to a smaller input tensor that is
    downsampled using the strides defined in 'stride'.

    Args:
        in_channels: The number of channels in the input tensor.
        out_channels: The number of channels in the output tensor.
        stride: The stride by which the input should be downsampled.
        anti_aliasing: Wether or not to apply anti-aliasing before downsampling.
        padding_factory: A factor for producing the padding blocks used in the model.

    Return:
        A projection module to project the input to the dimensions of the output.
    """
    if max(stride) == 1:
        if in_channels == out_channels:
            return nn.Identity()
        if len(stride) == 3:
            return nn.Conv3d(in_channels, out_channels, kernel_size=1)
        return nn.Conv2d(in_channels, out_channels, kernel_size=1)


    blocks = []

    if anti_aliasing:
        pad = tuple([1 if strd > 1 else 0 for strd in stride])
        filter_size = tuple([3 if strd > 1 else 1 for strd in stride])
        strd = (1,) * len(stride)
        blocks += [
            padding_factory(pad),
            BlurPool(in_channels, strd, filter_size)
        ]

    if len(stride) == 3:
        blocks.append(
            nn.Conv3d(in_channels, out_channels, kernel_size=stride, stride=stride)
        )
    else:
        blocks.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=stride, stride=stride)
        )
    return nn.Sequential(*blocks)


class Reflect(nn.Module):
    """
    Pad input by reflecting the input tensor.
    """
    def __init__(self, pad: Union[int, Tuple[int]]):
        """
        Instantiates a padding layer.

        Args:
            pad: N-tuple defining the padding added to the n-last dimensions
                of the tensor. If an int, the same padding will be added to the
                two last dimensions of the tensor.
        """
        super().__init__()
        if isinstance(pad, int):
            pad = (pad,) * 2

        full_pad = []
        for n_elems in pad:
            if isinstance(n_elems, (tuple, list)):
                full_pad += [n_elems[0], n_elems[1]]
            elif isinstance(n_elems, int):
                full_pad += [n_elems, n_elems]
            else:
                raise ValueError(
                    "Expected elements of pad tuple to be tuples of integers or integers. "
                    "Got %s.", type(n_elems)
                )

        full_pad = tuple(full_pad[::-1])
        self.pad = full_pad


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add padding to tensor.

        Args:
            x: The input tensor.

        Return:
            The padded tensor.
        """
        return nn.functional.pad(x, self.pad, "reflect")


class LayerNormFirst(nn.Module):
    """
    Layer norm performed along the first dimension.
    """

    def __init__(self, n_channels, eps=1e-5):
        """
        Args:
            n_channels: The number of channels in the input.
            eps: Epsilon added to variance to avoid numerical issues. """
        super().__init__()
        self.n_channels = n_channels
        self.scaling = nn.Parameter(torch.ones(n_channels), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(n_channels), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        """
        Apply normalization to x.
        """
        dtype = x.dtype
        mu = x.mean(1, keepdim=True)
        x_n = (x - mu).to(dtype=torch.float32)
        var = x_n.pow(2).mean(1, keepdim=True)
        x_n = x_n / torch.sqrt(var + self.eps)
        shape_ext = (self.n_channels,) + (1,) * (x_n.dim() - 2)
        x = self.scaling.reshape(shape_ext) * x_n.to(dtype=dtype) + self.bias.reshape(shape_ext)
        return x


class InvertedBottleneck(nn.Module):
    """
    Inverted-bottleneck block is used in MobileNet and Efficient net where it is referred
    to as MBConv
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
        self.act = nn.GELU()
        act = nn.GELU()

        expansion_factors = {
            0: 1,
            1: 4,
            2: 4,
            3: 5,
            4: 6,
            4: 6
        }
        expansion_factor = expansion_factors.get(stage_ind, 2)

        hidden_channels = out_channels * expansion_factor

        stride = (1, 1)
        if downsample is not None:
            if isinstance(downsample, int):
                downsample = (downsample,) * 2
            if max(downsample) > 1:
                stride = downsample

        self.projection = get_projection(
            in_channels,
            out_channels,
            stride=stride,
        )

        fused = 2 <= stage_ind

        blocks = []
        if not fused:
            blocks += [
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
                LayerNormFirst(hidden_channels),
                act
            ]

            blocks += [
                Reflect(1),
                nn.Conv2d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=stride,
                    groups=hidden_channels,
                ),
                LayerNormFirst(hidden_channels),
                act
            ]
        else:
            blocks += [
                Reflect(1),
                nn.Conv2d(
                    in_channels,
                    hidden_channels,
                    kernel_size=3,
                    stride=stride,
                ),
                LayerNormFirst(hidden_channels),
                act
            ]

        blocks += [
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
            LayerNormFirst(out_channels),
            act
        ]
        self.body = nn.Sequential(*blocks)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate input through layer.
        """
        shortcut = self.projection(x)

        ## Apply stochastic depth.
        #if self.stochastic_depth is not None and self.training:
        #    p = torch.rand(1)
        #    if p <= self.stochastic_depth:
        #        return shortcut + self.body(x)
        #    return shortcut

        return shortcut + self.body(x)


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention with relative position biases.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """
        Args:
            in_channels: The number of input channels.
            out_channels: The number of input channels.
            window_size: The size of the attention windows.
            num_heads: The number of attention heads.
            qkv_bias: Whether to include biases for the QKV mapping.
            attn_drop: Dropout fraction to apply to the attention.
            proj_drop: Dropout to apply to the projection.

        """
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        n_pos = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(num_heads, n_pos)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(in_channels, out_channels * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(out_channels, out_channels)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B, N, L, C = x.shape
        qkv = self.qkv(x).reshape(B, N, L, 3, self.num_heads, -1).permute(0, 1, 4, 2, 3, 5)
        C = qkv.shape[-1]
        q, k, v = torch.unbind(qkv, dim=-2)

        relative_position_bias = self.relative_position_bias_table[
            ..., self.relative_position_index.view(-1)
        ].view(1, self.num_heads, L, L)
        if mask is not None:
            relative_position_bias = relative_position_bias[None].repeat_interleave(B, 0).repeat_interleave(N, 1)
            mask = mask[:, :, None].repeat_interleave(self.num_heads, 2)
            relative_position_bias[mask] = -100.0

        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=relative_position_bias, dropout_p=self.attn_drop
        )

        x = self.proj(x.permute(0, 1, 3, 2, 4).reshape(B, N, L, -1))
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows.
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = (
        x.permute(0, 2, 4, 3, 5, 1).contiguous().view(B, H * W // window_size // window_size, window_size * window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse window partition.
    """
    B, N, L, C = windows.shape
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return x


class PatchMerging(nn.Module):
    """
    Downsamples a sequence of tokens representing a QUADRATIC input image by a factor of two by merging
    2 x 2 neighborhoods of patches.
    """

    def __init__(self, chans_in: int, chans_out: Optional[int] = None):
        """
        Args:
            chans_in: The number of features in the input tokens.
            chans_out: The number of features in the output tokens.
        """
        super().__init__()
        if chans_out is None:
            chans_out = 2 * chans_in
        self.chans_in = chans_in
        self.chans_out = chans_out
        self.proj = nn.Linear(4 * chans_in, chans_out, bias=False)
        self.norm = nn.LayerNorm(4 * chans_in)

    def forward(self, x):

        x0 = x[:, :, 0::2, 0::2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], 1)

        x = x.permute((0, 2, 3, 1))
        x = self.norm(x)
        x = self.proj(x)
        x = x.permute((0, 3, 1, 2))

        return x


class SwinAttention(nn.Module):
    """
    Swin Transformer Block applying window attention and shift.
    """
    num_heads = (4, 4, 8, 16, 24, 24)


    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stage_ind: int,
        block_ind: int,
        downsample: Optional[Tuple[int, int]] = None,
        window_size: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        mlp_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        """
        Args:
            in_channels: The number of channels in the input.
            out_channels: The number of channels in the input.
            stage_ind: The index defining the in which stage of the encoder this block is added.
            blocks_ind: The index defining the running number of blocks in this stage.
            window_size: The size of the attention window.
            shift_size: The shift to apply in the block.
            mlp_ratio: Factor by which to increase the channels to use for the MLP.
            qkv_bias: Whether or not to include a bias in the qkv projection.
            drop: Dropout applied in the MLP block..

        """
        super().__init__()
        num_heads = self.num_heads[stage_ind]
        image_size = 256 // 2 ** stage_ind
        image_size = (image_size,) * 2
        shift_size = 0 if block_ind % 2 == 0 else window_size // 2

        if downsample is not None:
            self.downsample = PatchMerging(in_channels, out_channels)
            in_channels = out_channels
        else:
            self.downsample = None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_size = image_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.image_size) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.image_size)

        self.norm1 = nn.LayerNorm(self.in_channels)
        self.attn = WindowAttention(
            self.in_channels,
            self.out_channels,
            window_size=[self.window_size, self.window_size],
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=mlp_drop,
        )

        if in_channels == out_channels:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(in_channels, out_channels)

        self.drop_path = nn.Identity() if drop_path <= 0.0 else nn.Dropout(drop_path)
        self.norm2 = nn.LayerNorm(self.out_channels)
        mlp_hidden_dim = int(self.out_channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.out_channels, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(mlp_hidden_dim, self.out_channels),
            nn.Dropout(mlp_drop),
        )

        if self.shift_size > 0:
            H, W = self.image_size
            img_mask = torch.zeros((H, W))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[h, w] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask[None, None], self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) != mask_windows.unsqueeze(2)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.downsample:
            x = self.downsample(x)

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(-2, -1)
            )
        else:
            shifted_x = x
        B, C = x.shape[:2]
        x = window_partition(shifted_x, self.window_size)

        H, W = self.image_size

        shortcut = self.projection(x)
        x = self.norm1(x)

        mask = self.attn_mask
        if mask is not None:
            mask = mask[None].repeat_interleave(B, dim=0)
        attn_windows = self.attn(x, mask=mask)

        attn_windows = attn_windows
        x = shortcut + self.drop_path(attn_windows)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = window_reverse(x, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(
                x,
                shifts=(self.shift_size, self.shift_size),
                dims=(-2, -1),
            )
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
                block_ind=block_ind,
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
                block_ind=block_ind,
            )
            blocks.append(block)
            current_channels = out_channels
        self.blocks = nn.Sequential(*blocks)

        self.upsample = None
        if upsample is not None:
            self.upsample = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )

    def forward(self, x: torch.Tensor, x_sc: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Args:
            x: The output from the previous stage.
            x_sc: The shortcut tensor from the corresponding encoder stage.

        Return:
            The output from the stage.
        """
        if self.upsample is not None:
            x = self.upsample(x)
        if x_sc is not None:
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

        for stage_idx, (stage_channels, stage_depth) in enumerate(
            zip(channels, depths)
        ):
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

        for stage_idx, (stage_channels, stage_depth) in enumerate(
            zip(decoder_channels, decoder_depths)
        ):
            stage = DecoderStage(
                in_channels=current_channels,
                out_channels=stage_channels,
                sc_channels=stage_channels,
                block_factory=block_factory,
                depth=stage_depth,
                upsample=(2, 2),
                stage_ind=len(channels) - 2 - stage_idx,
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
        block_name = self.block_factory.replace("Block", "").lower()

        # Create channel signature: input -> [stages] -> output
        stages_str = "x".join(map(str, self.channels))

        # Create depth signature
        if all(d == self.depths[0] for d in self.depths):
            # All depths are the same
            depth_str = f"d{self.depths[0]}"
        else:
            # Different depths per stage
            depth_str = "d" + "x".join(map(str, self.depths))

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


def compile_model(
    model_config_path: Union[str, Path],
    satrain_config_path: Union[str, Path],
) -> EncoderDecoder:
    """
    Load Encoder-Decoder Model.

    Args:
        model_path: Path of the trained model.

    Return:
        A tuple ``(model, satrfain_config)`` containing the loaded model and the SatRain dataset
        configuration used to train the model.
    """
    model_config_path = Path(model_config_path)
    model_config = EncoderDecoderConfig.from_toml_file(model_config_path)
    satrain_config = SatRainConfig.from_toml_file(satrain_config_path)
    block_factory = globals()[model_config.block_factory]
    model = EncoderDecoder(
        block_factory=block_factory,
        in_channels=satrain_config.num_features,
        channels=model_config.channels,
        depths=model_config.depths,
        out_channels=model_config.out_channels,
        bilinear=model_config.bilinear,
    )
    return model, satrain_config


def load_model(model_path: Union[str, Path]) -> EncoderDecoder:
    """
    Load Encoder-Decoder Model.

    Args:
        model_path: Path of the trained model.

    Return:
        A tuple ``(model, satrfain_config)`` containing the loaded model and the SatRain dataset
        configuration used to train the model.
    """
    model_path = Path(model_path)

    loaded = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
    state = loaded["state_dict"]
    # Remove model prefix for checkpoint files.
    if model_path.suffix == ".ckpt":
        state = {key[6:]: val for key, val in state.items()}

    satrain_config = SatRainConfig(**loaded["satrain_config"])
    model_config = EncoderDecoderConfig(**loaded["model_config"])
    block_factory = globals()[model_config.block_factory]
    # Create model
    model = EncoderDecoder(
        block_factory=block_factory,
        in_channels=satrain_config.num_features,
        channels=model_config.channels,
        depths=model_config.depths,
        out_channels=model_config.out_channels,
        bilinear=model_config.bilinear,
    )
    model.load_state_dict(state)
    return model, satrain_config
