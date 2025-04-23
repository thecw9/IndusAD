from typing import Tuple
import torch

import torch.nn as nn
from torch import Tensor

from src.networks.autoencoder import Decoder, Encoder
from src.networks.gan import Discriminator


class Generator(nn.Module):
    def __init__(
        self,
        in_channels=1,
        image_size=128,
        latent_dim=32,
        channels=[64, 128, 256, 512, 512],
        final_activation="sigmoid",
        residual: bool = True,
    ):
        super(Generator, self).__init__()
        self.encoder1 = Encoder(
            in_channels=in_channels,
            image_size=image_size,
            channels=channels,
            latent_dim=latent_dim,
            residual=residual,
        )
        self.decoder = Decoder(
            out_channels=in_channels,
            image_size=image_size,
            channels=channels[::-1],
            conv_input_size=self.encoder1.conv_output_size,
            final_activation=final_activation,
            residual=residual,
        )
        self.encoder2 = Encoder(
            in_channels=in_channels,
            image_size=image_size,
            channels=channels,
            latent_dim=latent_dim,
            residual=residual,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        latent_i = self.encoder1(x)
        x_recon = self.decoder(latent_i)
        latent_o = self.encoder2(x_recon)
        return x_recon, latent_i, latent_o


class Ganomaly(nn.Module):
    def __init__(
        self,
        in_channels=1,
        image_size=128,
        latent_dim=32,
        channels=[64, 128, 256, 512, 512],
        final_activation="sigmoid",
        residual: bool = False,
    ):
        super(Ganomaly, self).__init__()
        self.generator = Generator(
            in_channels=in_channels,
            image_size=image_size,
            latent_dim=latent_dim,
            channels=channels,
            final_activation=final_activation,
            residual=residual,
        )
        self.discriminator = Discriminator(
            in_channels=in_channels,
            image_size=image_size,
            channels=channels,
            residual=residual,
        )

        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)

    @staticmethod
    def weights_init(module: nn.Module) -> None:
        """Initialize DCGAN weights.

        Args:
            module (nn.Module): [description]
        """
        classname = module.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x_recon, latent_i, latent_o = self.generator(x)
        return x_recon, latent_i, latent_o


if __name__ == "__main__":
    model = Ganomaly()
    x = torch.rand(1, 1, 128, 128)
    x_recon, latent_i, latent_o = model(x)
    print(x_recon.shape, latent_i.shape, latent_o.shape)
