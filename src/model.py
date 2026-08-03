"""A small fully-convolutional U-Net for binary segmentation.

Unlike the dense-bottleneck U-Net in holey_segment, this one has no fully
connected layer, so a model trained on crops can be run on whole images of a
different size without retraining.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """(conv - norm - ReLU) x 2."""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
        depth: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depth = depth
        channels = [base_channels * 2 ** i for i in range(depth + 1)]

        self.input_block = ConvBlock(in_channels, channels[0], dropout)
        self.pool = nn.MaxPool2d(2)

        self.encoders = nn.ModuleList(
            ConvBlock(channels[i], channels[i + 1], dropout) for i in range(depth)
        )
        self.upsamples = nn.ModuleList(
            nn.ConvTranspose2d(channels[i + 1], channels[i], 2, stride=2)
            for i in reversed(range(depth))
        )
        self.decoders = nn.ModuleList(
            # 2 * channels[i] because of the concatenated skip connection
            ConvBlock(2 * channels[i], channels[i], dropout)
            for i in reversed(range(depth))
        )
        self.head = nn.Conv2d(channels[0], out_channels, 1)

    @property
    def size_multiple(self):
        """Input height/width must be a multiple of this."""
        return 2 ** self.depth

    def forward(self, x):
        skips = []
        x = self.input_block(x)
        for encoder in self.encoders:
            skips.append(x)
            x = encoder(self.pool(x))
        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = decoder(torch.cat([upsample(x), skip], dim=1))
        return self.head(x)  # logits


def build_model(base_channels=16, depth=3, dropout=0.0):
    return UNet(base_channels=base_channels, depth=depth, dropout=dropout)
