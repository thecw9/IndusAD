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
        residual: bool = False,
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
    ):
        super(Discriminator, self).__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.channels = channels

        modules = []
        for ch in channels:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, ch, kernel_size=4, stride=2, padding=1),
                    nn.InstanceNorm2d(ch),
                    nn.LeakyReLU(0.2),
                )
            )
            in_channels = ch

        self.conv = nn.Sequential(*modules)

        num_fc_in = int(torch.prod(torch.tensor(self.conv_output_size)))
        self.fc = nn.Sequential(
            nn.Linear(num_fc_in, 1),
        )

    def _get_conv_output_size(self) -> Tuple[int, int, int]:
        dummy_input = torch.zeros(
            1, self.in_channels, self.image_size, self.image_size
        ).to(next(self.parameters()).device)
        dummy_input = self.conv(dummy_input)
        return dummy_input[0].shape

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    @property
    def conv_output_size(self) -> Tuple[int, int, int]:
        return self._get_conv_output_size()
