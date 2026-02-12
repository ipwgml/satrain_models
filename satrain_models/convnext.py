"""
ConvNeXt model implementation for SatRain dataset.
Modified for regression with spatial output and UNet-style decoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    convnext_tiny, convnext_small, convnext_base, convnext_large,
    ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, 
    ConvNeXt_Base_Weights, ConvNeXt_Large_Weights
)
import logging

logger = logging.getLogger(__name__)


class UNetDecoder(nn.Module):
    """UNet-style decoder with skip connections for ConvNeXt."""
    def __init__(self, encoder_channels: int, decoder_channels: list, n_outputs: int = 1):
        super().__init__()
        
        # Decoder blocks that handle concatenated skip connections
        # encoder_channels is the bottleneck feature dimension
        self.decode4 = self._decode_block(encoder_channels + decoder_channels[0], decoder_channels[0])
        self.decode3 = self._decode_block(decoder_channels[0] + decoder_channels[1], decoder_channels[1])
        self.decode2 = self._decode_block(decoder_channels[1] + decoder_channels[2], decoder_channels[2])
        
        # Final decoder stage - no skip connection (stride 1 doesn't exist in ConvNeXt)
        self.decode1 = self._decode_block(decoder_channels[2], decoder_channels[3])
        
        # Final output layer
        self.final_conv = nn.Conv2d(decoder_channels[3], n_outputs, kernel_size=1)
        
    def _decode_block(self, in_channels: int, out_channels: int):
        """Create a decoder block with upsampling and convolutions."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip_connections):
        """
        Args:
            x: Features from encoder (deepest level)
            skip_connections: List of skip connections [stride_16, stride_8, stride_4]
                            (Note: ConvNeXt doesn't have stride_2 or stride_1)
        """
        # Decoder stage 4: stride 32 -> stride 16
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip_connections[0] is not None:
            skip = skip_connections[0]
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.decode4(x)
        
        # Decoder stage 3: stride 16 -> stride 8
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip_connections[1] is not None:
            skip = skip_connections[1]
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.decode3(x)
        
        # Decoder stage 2: stride 8 -> stride 4
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip_connections[2] is not None:
            skip = skip_connections[2]
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.decode2(x)
        
        # Decoder stage 1: stride 4 -> stride 2 -> stride 1 (no skip connection)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.decode1(x)
        
        # Final upsampling: stride 2 -> stride 1
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Final output
        out = self.final_conv(x)
        return out

class ConvNeXtRegression(nn.Module):
    """ConvNeXt with UNet-style decoder for regression with spatial output."""
    def __init__(self, base_model, n_channels: int, n_outputs: int = 1):
        super().__init__()
        
        # ConvNeXt structure: features contains the encoder stages
        self.features = base_model.features
        
        # Modify first conv layer for different input channels
        if n_channels != 3:
            original_conv = self.features[0][0]
            self.features[0][0] = nn.Conv2d(
                n_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=(original_conv.bias is not None)
            )
        
        # Extract skip connection info by analyzing the model architecture
        self._analyze_and_extract_skip_info(n_channels)
        
        # Log extracted information
        logger.info(f"Extracted skip indices: {self.skip_indices}")
        logger.info(f"Extracted skip channels: {self.skip_channels}")
        logger.info(f"Extracted skip strides: {self.skip_strides}")
        logger.info(f"Bottleneck channels: {self.bottleneck_channels}")
        
        # Define decoder output channels for each stage
        decoder_channels = [256, 128, 64, 32]
        
        # Adapter layers to match skip connection channels to decoder expectations
        self.skip_adapters = nn.ModuleList()
        for i, skip_ch in enumerate(self.skip_channels):
            self.skip_adapters.append(
                nn.Conv2d(skip_ch, decoder_channels[i], kernel_size=1)
            )
        
        # UNet-style decoder with proper channel dimensions
        self.decoder = UNetDecoder(self.bottleneck_channels, decoder_channels, n_outputs)
        
    def _analyze_and_extract_skip_info(self, n_channels: int):
        """Robustly extract skip connection info by analyzing feature map sizes."""
        self.skip_stride_dict = {}  # Maps stride -> (layer_idx, channels, spatial_size)
        
        # Create a dummy input to trace through the network
        dummy_input = torch.zeros(1, n_channels, 256, 256)
        input_height = dummy_input.shape[2]
        
        with torch.no_grad():
            x = dummy_input
            for idx, module in enumerate(self.features):
                x = module(x)
                
                # Calculate current stride based on spatial dimensions
                current_height = x.shape[2]
                current_stride = input_height // current_height
                
                # Store all occurrences, keyed by stride
                if current_stride in [1, 2, 4, 8, 16, 32]:
                    # Always overwrite to get the last layer at each stride
                    self.skip_stride_dict[current_stride] = {
                        'idx': idx,
                        'channels': x.shape[1],
                        'spatial_size': current_height
                    }
        
        # ConvNeXt architecture typically has: stride 4, 8, 16, 32
        # Extract skip connections in order: stride 16, 8, 4
        # (Note: stride 2 and 1 typically don't exist in ConvNeXt)
        self.skip_indices = []
        self.skip_channels = []
        self.skip_strides = []
        
        for target_stride in [16, 8, 4]:
            if target_stride in self.skip_stride_dict:
                info = self.skip_stride_dict[target_stride]
                self.skip_indices.append(info['idx'])
                self.skip_channels.append(info['channels'])
                self.skip_strides.append(target_stride)
            else:
                logger.warning(f"Could not find skip connection at stride {target_stride}")
        
        # Store bottleneck channels (highest stride, usually 32)
        if 32 in self.skip_stride_dict:
            self.bottleneck_channels = self.skip_stride_dict[32]['channels']
        else:
            # Fallback to the last available stride
            max_stride = max(self.skip_stride_dict.keys())
            self.bottleneck_channels = self.skip_stride_dict[max_stride]['channels']
            logger.warning(f"No stride 32 found, using stride {max_stride} for bottleneck")
        
    def forward(self, x):
        # Get input size for final upsampling
        input_size = (x.shape[2], x.shape[3])
        
        # Extract features with skip connections
        skip_connections = [None, None, None]  # Only 3 skip connections for ConvNeXt
        
        for idx, module in enumerate(self.features):
            x = module(x)
            
            # Collect skip connections from key layers
            if idx in self.skip_indices:
                try:
                    skip_pos = self.skip_indices.index(idx)
                    if skip_pos < 3:  # Only 3 skip connections
                        adapted = self.skip_adapters[skip_pos](x)
                        skip_connections[skip_pos] = adapted
                except (ValueError, IndexError) as e:
                    logger.error(f"Error processing skip connection at layer {idx}: {e}")
                    raise
        
        features = x
        
        # Apply decoder with skip connections
        try:
            out = self.decoder(features, skip_connections)
        except RuntimeError as e:
            logger.error(f"Error in decoder: {e}")
            logger.error(f"Features shape: {features.shape}")
            logger.error(f"Skip connections shapes: {[s.shape if s is not None else None for s in skip_connections]}")
            raise
        
        # Final upsampling to input resolution if needed
        out = F.interpolate(out, size=input_size, mode='bilinear', align_corners=False)
        
        return out


def create_convnext(n_channels: int, n_outputs: int = 1, 
                    model_size: str = 'tiny', pretrained: bool = True) -> nn.Module:
    """
    Create a ConvNeXt model with UNet decoder for regression with spatial output.
    
    Args:
        n_channels: Number of input channels
        n_outputs: Number of output channels (default: 1 for regression)
        model_size: Size of the model ('tiny', 'small', 'base', 'large')
        pretrained: Whether to use pretrained weights
    
    Returns:
        Modified ConvNeXt model with UNet decoder
        
    Raises:
        ValueError: If model_size is not valid
    """
    # Create base model with appropriate weights
    if model_size == 'tiny':
        base_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_size == 'small':
        base_model = convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_size == 'base':
        base_model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None)
    elif model_size == 'large':
        base_model = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1 if pretrained else None)
    else:
        raise ValueError(f"model_size must be 'tiny', 'small', 'base', or 'large', got {model_size}")

    # Create regression model with UNet decoder
    model = ConvNeXtRegression(
        base_model=base_model,
        n_channels=n_channels,
        n_outputs=n_outputs
    )
    
    logger.info(f"Created ConvNeXt model (size={model_size}, n_channels={n_channels}, n_outputs={n_outputs})")
    return model
