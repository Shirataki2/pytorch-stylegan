import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_layer import FullyConnect


class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise):
        if noise is None:
            noise = torch.randn(
                x.size(0),
                1,
                x.size(2),
                x.size(3),
                device=x.device,
                dtype=x.dtype
            )
        return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)


class ApplyStyle(nn.Module):
    def __init__(self, latent_size, channels, use_wscale):
        super().__init__()
        self.fc = FullyConnect(
            latent_size,
            channels * 2,
            gain=1,
            use_wscale=use_wscale
        )

    def forward(self, x, latent):
        style = self.fc(latent).view([-1, 2, x.size(1), 1, 1])
        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x
