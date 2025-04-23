from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary

from src.networks.autoencoder import Decoder, Encoder
from src.networks.residual import ResidualBlock


class Generator(nn.Module):
    def __init__(
        self,
        out_channels=1,
        image_size=128,
        latent_dim=32,
        channels=[64, 128, 256, 512, 512],
        conv_input_size=(512, 4, 4),  # (channels, size, size
        final_activation="sigmoid",
        residual: bool = True,
    ):
        super(Generator, self).__init__()

        self.decoder = Decoder(
            out_channels=out_channels,
            image_size=image_size,
            latent_dim=latent_dim,
            channels=channels,
            conv_input_size=conv_input_size,
            final_activation=final_activation,
            residual=residual,
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.decoder(z)


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels=1,
        image_size=128,
        channels=[64, 128, 256, 512, 512],
        residual: bool = True,
    ):
        super(Discriminator, self).__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            image_size=image_size,
            latent_dim=1,
            channels=channels,
            residual=residual,
        )

        self.activation = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.encoder(x))

    @property
    def conv_output_size(self) -> Tuple[int, int, int]:
        return self.encoder.conv_output_size
