"""
어탠션 혹은 사전학습 사용 X
32 -> 16 -> 8 -> 16 -> 32 U-Net
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class TimestepEmbedding(nn.Module):
    def __init__(self, width):
        super().__init__()
        if width < 4 or width % 2:
            raise ValueError("Embedding width must be even and >= 4")
        frequencies = torch.exp(-math.log(10000) * torch.arange(width // 2) / (width // 2 - 1))
        self.register_buffer("frequencies", frequencies)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.SiLU(), nn.Linear(width * 4, width * 4))

    def forward(self, timesteps):
        angles = timesteps.float()[:, None] * self.frequencies[None, :]
        return self.mlp(torch.cat((angles.sin(), angles.cos()), dim=1))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(math.gcd(8, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_projection = nn.Linear(time_channels, out_channels)
        self.norm2 = nn.GroupNorm(math.gcd(8, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_embedding):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        # ResidualBlock
        return self.skip(x) + self.conv2(F.silu(self.norm2(h)))


class UNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, channel_multipliers=(1, 2, 4)):
        super().__init__()
        if len(channel_multipliers) != 3:
            raise ValueError("Exactly three resolutions are required")
        c0, c1, c2 = [base_channels * m for m in channel_multipliers]
        time_channels = base_channels * 4
        self.time_embedding = TimestepEmbedding(base_channels)
        self.input_conv = nn.Conv2d(in_channels, c0, 3, padding=1)
        self.encoder32 = ResidualBlock(c0, c0, time_channels)
        self.down16 = nn.Conv2d(c0, c1, 4, stride=2, padding=1)
        self.encoder16 = ResidualBlock(c1, c1, time_channels)
        self.down8 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)
        self.middle1 = ResidualBlock(c2, c2, time_channels)
        self.middle2 = ResidualBlock(c2, c2, time_channels)
        self.up16 = nn.Conv2d(c2, c1, 3, padding=1)
        self.decoder16 = ResidualBlock(c1 * 2, c1, time_channels)
        self.up32 = nn.Conv2d(c1, c0, 3, padding=1)
        self.decoder32 = ResidualBlock(c0 * 2, c0, time_channels)
        self.output = nn.Sequential(nn.GroupNorm(math.gcd(8, c0), c0), nn.SiLU(),
                                    nn.Conv2d(c0, in_channels, 3, padding=1))

    def forward(self, x, timesteps):
        if x.ndim != 4 or x.shape[-2:] != (32, 32):
            raise ValueError("Expected (N, C, 32, 32)")
        if timesteps.shape != (x.shape[0],):
            raise ValueError("One timestep is required per image")
        embedding = self.time_embedding(timesteps)
        skip32 = self.encoder32(self.input_conv(x), embedding)
        skip16 = self.encoder16(self.down16(skip32), embedding)
        h = self.middle2(self.middle1(self.down8(skip16), embedding), embedding)
        h = self.up16(F.interpolate(h, scale_factor=2, mode="nearest"))
        # skip-connection
        h = self.decoder16(torch.cat((h, skip16), dim=1), embedding)
        h = self.up32(F.interpolate(h, scale_factor=2, mode="nearest"))
        return self.output(self.decoder32(torch.cat((h, skip32), dim=1), embedding))
