from typing import Tuple

import torch
import torch.nn as nn
from src.utils import reparameterize
from src.networks.residual import ResidualBlock


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        latent_dim=512,
        channels=(64, 128, 256, 512, 512, 512),
        image_size=256,
    ):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.image_size = image_size
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 5, 1, 2, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )

        in_channels = channels[0]
        for channel in channels[1:]:
            self.main.append(
                nn.Sequential(
                    ResidualBlock(in_channels, channel),
                    nn.AvgPool2d(2),
                )
            )
            in_channels = channel

        self.main.append(ResidualBlock(in_channels, in_channels))
        num_fc_features = int(torch.prod(torch.tensor(self.conv_output_size)))
        self.fc = nn.Linear(num_fc_features, 2 * latent_dim)

    def calc_conv_output_size(self):
        dummy_input = torch.zeros(
            1, self.in_channels, self.image_size, self.image_size
        ).to(next(self.parameters()).device)
        dummy_input = self.main(dummy_input)
        return dummy_input[0].shape

    @property
    def conv_output_size(self):
        return self.calc_conv_output_size()

    def forward(self, x):
        y = self.main(x).view(x.size(0), -1)
        y = self.fc(y)
        mu, logvar = y.chunk(2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels=3,
        latent_dim=512,
        channels=(512, 512, 512, 256, 128, 64),
        image_size=256,
        conv_input_size=(512, 4, 4),
        final_activation="none",
    ):
        super(Decoder, self).__init__()
        self.cdim = out_channels
        self.image_size = image_size
        self.conv_input_size = conv_input_size

        num_fc_features = int(torch.prod(torch.tensor(self.conv_input_size)))
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, num_fc_features),
            nn.ReLU(True),
        )

        in_channel = channels[0]
        self.main = nn.Sequential()
        for channel in channels:
            self.main.append(
                nn.Sequential(
                    ResidualBlock(in_channel, channel),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                )
            )
            in_channel = channel

        self.main.append(ResidualBlock(in_channel, in_channel))
        self.main.add_module("predict", nn.Conv2d(in_channel, out_channels, 5, 1, 2))

        if final_activation == "sigmoid":
            self.main.add_module("final", nn.Sigmoid())
        elif final_activation == "tanh":
            self.main.add_module("final", nn.Tanh())
        elif final_activation == "none":
            pass
        else:
            raise ValueError("final_activation must be 'sigmoid' or 'tanh'")

    def forward(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc(z)
        y = y.view(z.size(0), *self.conv_input_size)
        y = self.main(y)
        return y


class LRAAD(nn.Module):
    def __init__(
        self,
        in_channels=3,
        latent_dim=512,
        channels=(64, 128, 256, 512, 512, 512),
        image_size=256,
        final_activation="sigmoid",
    ):
        super(LRAAD, self).__init__()

        self.zdim = latent_dim

        self.encoder = Encoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            channels=channels,
            image_size=image_size,
        )

        self.decoder = Decoder(
            out_channels=in_channels,
            latent_dim=latent_dim,
            channels=channels[::-1],
            image_size=image_size,
            conv_input_size=self.encoder.conv_output_size,
            final_activation=final_activation,
        )

    def forward(self, x, deterministic=False):
        mu, logvar = self.encode(x)
        if deterministic:
            z = mu
        else:
            z = reparameterize(mu, logvar)
        rec = self.decode(z)
        return mu, logvar, z, rec

    def sample(self, z):
        y = self.decode(z)
        return y

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):
        y = self.decoder(z)
        return y


if __name__ == "__main__":
    pass
