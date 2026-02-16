"""
satrain_models.fully_connected
==============================

Provides an implementation of a basic PyTorch Fully Connected Network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
