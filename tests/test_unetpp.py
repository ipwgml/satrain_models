import torch

from satrain_models import unetpp, unet
from satrain_models.encoder_decoder import ResNeXtBlock

def test_unetpp():
    """
    Test Unet++ architecture with simple double-conv block.
    """
    unet = unetpp.DenseEncoderDecoder(
        ResNeXtBlock,
        in_channels=13,
        channels=[32, 64, 128],
        depths=[1, 1, 1],
        out_channels=1,
        bilinear=True
    )

    x = torch.rand(1, 13, 128, 128)
    with torch.no_grad():
        y = unet(x)

    assert y.shape == (1, 1, 128, 128)
