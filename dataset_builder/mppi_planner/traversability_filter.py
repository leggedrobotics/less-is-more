import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class TraversabilityFilter(nn.Module):
    """Pretrained traversability CNN ported from elevation_mapping_cupy.

    Source: https://github.com/leggedrobotics/elevation_mapping_cupy/blob/main/elevation_mapping_cupy/script/elevation_mapping_cupy/traversability_filter.py
    """

    def __init__(self, w1, w2, w3, w_out, use_bias=False):
        super(TraversabilityFilter, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, dilation=1, padding=3, bias=use_bias)
        self.conv2 = nn.Conv2d(1, 4, 3, dilation=2, padding=3, bias=use_bias)
        self.conv3 = nn.Conv2d(1, 4, 3, dilation=3, padding=3, bias=use_bias)
        self.conv_out = nn.Conv2d(12, 1, 1, bias=use_bias)

        self.conv1.weight = nn.Parameter(torch.from_numpy(w1).float())
        self.conv2.weight = nn.Parameter(torch.from_numpy(w2).float())
        self.conv3.weight = nn.Parameter(torch.from_numpy(w3).float())
        self.conv_out.weight = nn.Parameter(torch.from_numpy(w_out).float())

    def forward(self, elevation):
        elevation = elevation.unsqueeze(0)
        out1 = self.conv1(elevation)
        out2 = self.conv2(elevation)
        out3 = self.conv3(elevation)

        out1 = out1[:, :, 2:-2, 2:-2]
        out2 = out2[:, :, 1:-1, 1:-1]
        out = torch.cat((out1, out2, out3), dim=1)
        out = self.conv_out(out.abs())
        return torch.exp(-out).squeeze()


def get_filter_torch(device: str = "cuda") -> TraversabilityFilter:
    data = np.load(Path(__file__).parent / "weights.npz")
    return (
        TraversabilityFilter(
            data["conv1_weight"],
            data["conv2_weight"],
            data["conv3_weight"],
            data["conv_final_weight"],
        )
        .to(device)
        .eval()
    )
