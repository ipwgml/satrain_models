import pytest
import torch

from satrain_models import UNet, create_unet


class TestUNet:
    """Test suite for UNet model."""

    def test_unet_initialization(self):
        """Test UNet model initialization with default parameters."""
        model = UNet(n_channels=3, n_outputs=1)

        assert model.n_channels == 3
        assert model.n_outputs == 1
        assert model.bilinear == False

        # Check that all layers are properly initialized
        assert hasattr(model, "inc")
        assert hasattr(model, "down1")
        assert hasattr(model, "down2")
        assert hasattr(model, "down3")
        assert hasattr(model, "down4")
        assert hasattr(model, "up1")
        assert hasattr(model, "up2")
        assert hasattr(model, "up3")
        assert hasattr(model, "up4")
        assert hasattr(model, "outc")

    def test_unet_initialization_bilinear(self):
        """Test UNet model initialization with bilinear upsampling."""
        model = UNet(n_channels=1, n_outputs=5, bilinear=True)

        assert model.n_channels == 1
        assert model.n_outputs == 5
        assert model.bilinear == True

    def test_unet_forward_pass_default(self):
        """Test UNet forward pass with default parameters."""
        model = UNet(n_channels=3, n_outputs=1)
        model.eval()

        # Test with different input sizes
        batch_sizes = [1, 2]
        input_sizes = [(256, 256), (128, 128), (64, 64)]

        for batch_size in batch_sizes:
            for h, w in input_sizes:
                x = torch.randn(batch_size, 3, h, w)

                with torch.no_grad():
                    output = model(x)

                # Check output shape
                assert output.shape == (batch_size, 1, h, w)
                assert output.dtype == torch.float32

    def test_unet_forward_pass_multichannel(self):
        """Test UNet forward pass with multiple input and output channels."""
        model = UNet(n_channels=5, n_outputs=10)
        model.eval()

        x = torch.randn(2, 5, 128, 128)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 10, 128, 128)

    def test_unet_forward_pass_bilinear(self):
        """Test UNet forward pass with bilinear upsampling."""
        model = UNet(n_channels=3, n_outputs=1, bilinear=True)
        model.eval()

        x = torch.randn(1, 3, 128, 128)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 1, 128, 128)

    def test_create_unet_function(self):
        """Test create_unet utility function."""
        # Test with default parameters
        model1 = create_unet()
        assert isinstance(model1, UNet)
        assert model1.n_channels == 3
        assert model1.n_outputs == 1
        assert model1.bilinear == False

        # Test with custom parameters
        model2 = create_unet(n_channels=1, n_outputs=5, bilinear=True)
        assert isinstance(model2, UNet)
        assert model2.n_channels == 1
        assert model2.n_outputs == 5
        assert model2.bilinear == True

    def test_unet_gradients(self):
        """Test that gradients flow through the model properly."""
        model = UNet(n_channels=3, n_outputs=1)
        model.train()

        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        output = model(x)

        # Compute a simple loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Check that gradients exist for model parameters
        for param in model.parameters():
            assert param.grad is not None

        # Check that input gradients exist
        assert x.grad is not None

    def test_unet_parameter_count(self):
        """Test that UNet has reasonable parameter count."""
        model = UNet(n_channels=3, n_outputs=1)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # UNet should have a significant number of parameters (>1M for standard config)
        assert total_params > 1_000_000
        assert (
            trainable_params == total_params
        )  # All parameters should be trainable by default

    def test_unet_different_input_sizes(self):
        """Test UNet with various input sizes to ensure it handles different dimensions."""
        model = UNet(n_channels=3, n_outputs=1)
        model.eval()

        # Test various sizes that are powers of 2 (works well with UNet architecture)
        sizes = [32, 64, 128, 256, 512]

        for size in sizes:
            x = torch.randn(1, 3, size, size)

            with torch.no_grad():
                output = model(x)

            assert output.shape == (1, 1, size, size)

    @pytest.mark.parametrize(
        "n_channels,n_outputs,bilinear",
        [
            (1, 1, False),
            (3, 1, False),
            (1, 3, False),
            (3, 3, False),
            (1, 1, True),
            (3, 1, True),
            (1, 3, True),
            (3, 3, True),
            (5, 10, False),
            (5, 10, True),
        ],
    )
    def test_unet_parametrized(self, n_channels, n_outputs, bilinear):
        """Parametrized test for various UNet configurations."""
        model = UNet(n_channels=n_channels, n_outputs=n_outputs, bilinear=bilinear)
        model.eval()

        x = torch.randn(1, n_channels, 128, 128)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, n_outputs, 128, 128)
