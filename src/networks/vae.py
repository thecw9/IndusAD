from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary

from src.networks.autoencoder import Decoder
from src.networks.residual import ResidualBlock


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        image_size=128,
        latent_dim=32,
        channels=[64, 128, 256, 512, 512],
        residual: bool = True,
    ):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.channels = channels

        modules = []
        for channel in channels:
            if residual:
                modules.append(
                    nn.Sequential(
                        ResidualBlock(in_channels, channel),
                        nn.AvgPool2d(2),
                    )
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, channel, 4, 2, 1),
                        nn.BatchNorm2d(channel),
                        nn.LeakyReLU(0.2),
                    )
                )
            in_channels = channel
        self.main = nn.Sequential(*modules)

        self.conv_output_size = self.calc_conv_output_size()
        num_fc_in = int(torch.prod(torch.tensor(self.conv_output_size)))
        self.fc = nn.Linear(num_fc_in, latent_dim * 2)

    def calc_conv_output_size(self) -> torch.Size:
        dummy_input = torch.zeros(1, self.in_channels, self.image_size, self.image_size)
        dummy_input = self.main(dummy_input)
        return dummy_input[0].shape

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        mu, logvar = x.chunk(2, dim=1)
        return mu, logvar


class VAE(nn.Module):
    def __init__(
        self,
        in_channels=1,
        image_size=128,
        latent_dim=32,
        channels=[64, 128, 256, 512, 512],
        final_activation="sigmoid",
        residual: bool = True,
    ):
        super(VAE, self).__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            image_size=image_size,
            latent_dim=latent_dim,
            channels=channels,
            residual=residual,
        )
        self.decoder = Decoder(
            out_channels=in_channels,
            image_size=image_size,
            latent_dim=latent_dim,
            channels=channels[::-1],
            conv_input_size=self.encoder.conv_output_size,
            final_activation=final_activation,
            residual=residual,
        )

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, z, mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


if __name__ == "__main__":
    in_channels = 1
    image_size = 128
    latent_dim = 32
    channels = [64, 128, 256, 512, 512]
    encoder = Encoder(in_channels, image_size, latent_dim, channels)
    summary(encoder, (in_channels, image_size, image_size), device="cpu")
    print(encoder)

    vae = VAE(in_channels, image_size, latent_dim, channels)
    summary(vae, (in_channels, image_size, image_size), device="cpu")
