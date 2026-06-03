"""
satrain_models.swinunet
=======================

Provides an implementation of a SwinUnet using Swin Transformer blocks.
"""

from math import sqrt
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """
    Patch embedding layer that splits images into patches and encodes them into tokens
    representing a different image patch.
    """

    def __init__(self, patch_size: int = 4, chans_in: int = 3, embed_dim: int = 96):
        super().__init__()
        self.proj = nn.Conv2d(
            chans_in, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, n_t, C]
        x = self.norm(x)
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
        B, L, _ = x.shape
        H = W = int(sqrt(L))
        x = x.view(B, H, W, self.chans_in)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * self.chans_in)

        x = self.norm(x)
        x = self.proj(x)

        return x


class PatchExpand(nn.Module):
    """
    Patch expanding layer for upsampling.
    """
    def __init__(
            self,
            chans_in: int,
    ):
        """
        Args:
            chans_in: The number of input channels.
        """
        super().__init__()
        self.chans_in = chans_in
        self.expand = nn.Linear(chans_in, 2 * chans_in, bias=False)
        self.norm = nn.LayerNorm(chans_in // 2)

    def forward(self, x):
        x = self.expand(x)
        B, L, C = x.shape
        H = W = int(sqrt(L))
        x = x.view(B, H, W, C)
        x = (
            x.view(B, H, W, 2, 2, C // 4)
            .permute(0, 1, 4, 2, 3, 5)
            .contiguous()
            .view(B, 2 * H, 2 * W, C // 4)
        )
        x = x.view(B, -1, C // 4)
        x = self.norm(x)

        return x


class FinalPatchExpand_X4(nn.Module):
    """
    Final patch expanding layer for 4x upsampling.
    """

    def __init__(self, input_resolution, dim, dim_scale=4):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = nn.LayerNorm(self.output_dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = (
            x.view(
                B,
                H,
                W,
                self.dim_scale,
                self.dim_scale,
                C // (self.dim_scale**2),
            )
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(
                B,
                self.dim_scale * H,
                self.dim_scale * W,
                C // (self.dim_scale**2),
            )
        )
        x = x.view(B, -1, C // (self.dim_scale**2))
        x = self.norm(x)

        return x


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention with relative position biases.
    """

    def __init__(
        self,
        chans_in: int,
        window_size: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.chans_in = chans_in
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = chans_in // num_heads
        self.scale = head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
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

        self.qkv = nn.Linear(chans_in, chans_in * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(chans_in, chans_in)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows.
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse window partition.
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block applying window attention and shift.
    """

    def __init__(
        self,
        chans_in: int,
        image_size: Tuple[int, int],
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        mlp_drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        """
        Args:
            chans_in: The number of channels in the input.
            image_size: The size of the input image.
            num_heads: The number of attention heads.
            window_size: The size of the attention window.
            shift_size: The shift to apply in the block.
            mlp_ratio: Factor by which to increase the channels to use for the MLP.
            qkv_bias: Whether or not to include a bias in the qkv projection.
            drop: Dropout applied in the MLP block..

        """
        super().__init__()
        self.chans_in = chans_in
        self.image_size = image_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.image_size) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.image_size)

        self.norm1 = nn.LayerNorm(chans_in)
        self.attn = WindowAttention(
            chans_in,
            window_size=[self.window_size, self.window_size],
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=mlp_drop,
        )

        self.drop_path = nn.Identity() if drop_path <= 0.0 else nn.Dropout(drop_path)
        self.norm2 = nn.LayerNorm(chans_in)
        mlp_hidden_dim = int(chans_in * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(chans_in, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_drop),
            nn.Linear(mlp_hidden_dim, chans_in),
            nn.Dropout(mlp_drop),
        )

        if self.shift_size > 0:
            H, W = self.image_size
            img_mask = torch.zeros((1, H, W, 1))
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
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = self.image_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class UNetStage(nn.Module):
    """
    A stage in the SWinUNet consisting of several Swin block and a downsampling layer.

    Conserves the number of channels in the input if no downsampling is applied. Otherwise the number of
    channels is doubled.
    """
    def __init__(
        self,
        chans_in: int,
        image_size: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        mlp_drop: float = 0.0,
        attn_drop: float =0.0,
        drop_path: float = 0.0,
        downsample: Callable[[int], nn.Module] = None,
    ):
        """
        Args:
            chans_in: The number of channels/features in the input tensor.
            image_size: The size of the input miage.
            depth: The number of Swin blocks in the stage.
            num_heads: The number of heads in each Swin block.
            window_size: The size of the attention window.
            mlp_ratio: The fractional increase in the number of channels in the MLP block.
            qkv_bias: Whether or not to include a bias in the attention block.
            mlp_drop: The dropout applied in the MLP.
            attn_drop: The dropout applied in the attention block.
            drop_path: The drop path to apply.
            downsample: Callable to create a down- or up-sampling module at the end of the stage.
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    chans_in=chans_in,
                    image_size=image_size,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    mlp_drop=mlp_drop,
                    attn_drop=attn_drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                )
                for i in range(depth)
            ]
        )

        if downsample is not None:
            self.downsample = downsample(chans_in)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagate tensor through stage.
        """
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x




class SwinUnet(nn.Module):
    """
    A Swin Transformer based UNet for image segmentation.
    """

    def __init__(
        self,
        n_channels=3,
        n_outputs=1,
        img_size=256,
        patch_size=4,
        embed_dim=96,
        depths=[2, 2, 2, 2],
        num_heads=[3, 6, 12, 24],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        bilinear=False,
    ):
        super(SwinUnet, self).__init__()
        self.n_channels = n_channels
        self.n_outputs = n_outputs
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.bilinear = bilinear

        patches_resolution = [img_size // patch_size, img_size // patch_size]
        self.patches_resolution = patches_resolution

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            chans_in=n_channels,
            embed_dim=embed_dim,
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = UNetStage(
                chans_in=int(embed_dim * 2**i_layer),
                image_size=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                mlp_drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                downsample=(PatchMerging if (i_layer < self.num_layers - 1) else None),
            )
            self.layers.append(layer)

        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = (
                nn.Linear(
                    2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                )
                if i_layer > 0
                else nn.Identity()
            )
            if i_layer == 0:
                layer_up = PatchExpand(
                    chans_in=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                )
            else:
                layer_up = UNetStage(
                    chans_in=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    image_size=(
                        patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                        patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer)),
                    ),
                    depth=depths[(self.num_layers - 1 - i_layer)],
                    num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    mlp_drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[
                        sum(depths[: (self.num_layers - 1 - i_layer)]) : sum(
                            depths[: (self.num_layers - 1 - i_layer) + 1]
                        )
                    ],
                    downsample=(
                        PatchExpand if (i_layer < self.num_layers - 1) else None
                    ),
                )
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = nn.LayerNorm(int(embed_dim * 2 ** (self.num_layers - 1)))
        self.norm_up = nn.LayerNorm(self.embed_dim)

        self.up = FinalPatchExpand_X4(
            input_resolution=(img_size // patch_size, img_size // patch_size),
            dim_scale=4,
            dim=embed_dim,
        )
        self.output = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=n_outputs,
            kernel_size=1,
            bias=False,
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)

        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[self.num_layers - 1 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)

        x = self.up(x)
        H, W = self.patches_resolution
        B, L, C = x.shape
        x = x.view(B, 4 * H, 4 * W, -1)
        x = x.permute(0, 3, 1, 2)
        x = self.output(x)

        return x


def create_swinunet(
    n_channels=3,
    n_outputs=1,
    img_size=256,
    patch_size=4,
    embed_dim=96,
    depths=[2, 2, 2, 2],
    num_heads=[3, 6, 12, 24],
    window_size=8,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    bilinear=True,
):
    """Create a SwinUnet model.

    Args:
        n_channels (int): Number of input channels. Default: 3
        n_outputs (int): Number of output channels. Default: 1
        img_size (int): Input image size. Default: 256
        patch_size (int): Patch size. Default: 4
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (list): Depth of each Swin Transformer layer. Default: [2, 2, 2, 2]
        num_heads (list): Number of attention heads in different layers. Default: [3, 6, 12, 24]
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        bilinear (bool): Whether to use bilinear upsampling. Default: False (unused but kept for compatibility)

    Returns:
        SwinUnet: SwinUnet model instance
    """
    return SwinUnet(
        n_channels=n_channels,
        n_outputs=n_outputs,
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        drop_path_rate=drop_path_rate,
        bilinear=bilinear,
    )
