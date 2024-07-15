import torch
import torch.nn as nn
from torchsummary import summary
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
        for c in channels:
            if residual:
                modules.append(
                    nn.Sequential(
                        ResidualBlock(in_channels, c),
                        nn.AvgPool2d(2),
                    ),
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, c, 4, 2, 1),
                        nn.BatchNorm2d(c),
                        nn.LeakyReLU(0.2),
                    ),
                )
            in_channels = c

        self.main = nn.Sequential(*modules)

        num_fc_in = int(torch.prod(torch.tensor(self.conv_output_size)))
        self.fc = nn.Linear(num_fc_in, latent_dim)

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
        x = self.main(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels=1,
        image_size=128,
        latent_dim=32,
        channels=[512, 512, 256, 128, 64],
        conv_input_size=(512, 4, 4),  # (channels, size, size
        final_activation="sigmoid",
        residual: bool = True,
    ):
        super(Decoder, self).__init__()
        self.out_channels = out_channels
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.channels = channels
        self.conv_input_size = conv_input_size
        self.final_activation = final_activation

        # check conv_input_size
        assert (
            conv_input_size[1] == conv_input_size[2]
        ), "conv_input_size must be square"
        assert (
            conv_input_size[0] == channels[0]
        ), "conv_input_size[0] must be equal to channels[0]"

        num_fc_in = latent_dim
        num_fc_out = 1
        for c in conv_input_size:
            num_fc_out *= c

        self.fc = nn.Sequential(
            nn.Linear(num_fc_in, num_fc_out),
            nn.ReLU(inplace=True),
        )

        size = conv_input_size[1]
        inchannel = conv_input_size[0]

        modules = []
        for c in channels:
            if residual:
                modules.append(
                    nn.Sequential(
                        ResidualBlock(inchannel, c),
                        nn.Upsample(
                            scale_factor=2, mode="bilinear", align_corners=False
                        ),
                    ),
                )
            else:
                modules.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(inchannel, c, 5, 2, 2, output_padding=1),
                        nn.BatchNorm2d(c),
                        nn.ReLU(inplace=True),
                    ),
                )
            inchannel, size = c, size * 2
        self.main = nn.Sequential(*modules)

        if final_activation == "sigmoid":
            self.main.add_module(
                "final",
                nn.Sequential(
                    nn.ConvTranspose2d(inchannel, out_channels, 5, 1, 2),
                    nn.Sigmoid(),
                ),
            )
        elif final_activation == "tanh":
            self.main.add_module(
                "final",
                nn.Sequential(
                    nn.ConvTranspose2d(inchannel, out_channels, 5, 1, 2),
                    nn.Tanh(),
                ),
            )
        elif final_activation == "none":
            self.main.add_module(
                "final",
                nn.ConvTranspose2d(inchannel, out_channels, 5, 1, 2),
            )
        else:
            raise ValueError("final_activation must be 'sigmoid' or 'tanh'")

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), *self.conv_input_size)
        x = self.main(x)
        return x


class Autoencoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        image_size=128,
        latent_dim=32,
        channels=[64, 128, 256, 512, 512],
        final_activation="sigmoid",
        residual: bool = True,
    ):
        super(Autoencoder, self).__init__()
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
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


if __name__ == "__main__":
    in_channels = 3
    image_size = 512
    encoder = Encoder(
        in_channels=in_channels,
        image_size=image_size,
        latent_dim=32,
        channels=[64, 128, 256, 512, 512],
    )
    summary(encoder, (in_channels, image_size, image_size), device="cpu")
    print(encoder)

    decoder = Decoder(
        out_channels=in_channels,
        image_size=image_size,
        latent_dim=32,
        channels=[512, 512, 256, 128, 64],
        conv_input_size=encoder.conv_output_size,
    )
    summary(decoder, (32,), device="cpu")

    autoencoder = Autoencoder(
        in_channels=in_channels,
        image_size=image_size,
        latent_dim=32,
        channels=[64, 128, 256, 512, 512],
    )
    summary(autoencoder, (in_channels, image_size, image_size), device="cpu")
