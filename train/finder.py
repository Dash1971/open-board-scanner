"""BoardFinder: a small network that finds the four corners of a chess board.

Input: float32 RGB, N x 3 x 256 x 256, values 0..255: the whole photo or
screenshot, shrunk to fit and padded to a square. Outputs:

- corners, N x 4 x 2: x and y of the board's top-left, top-right,
  bottom-right and bottom-left corners as they appear in the image, as
  fractions of the input's width and height;
- one "a whole board is in view" logit per image;
- the heatmap logits the corners were read from, N x 4 x 64 x 64.

The backbone is the same narrow ResNet as BoardNet. Its 8 x 8 features are
brought back up to 64 x 64 through a small feature pyramid, and each corner
is the mean position under its heatmap, which gives sub-cell accuracy and
needs no code outside the model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .model import BasicBlock, _conv_bn_relu

GRID = 64


class BoardFinder(nn.Module):
    # (channels, blocks, stride) per stage, running at 64, 32, 16 and 8 cells.
    STAGES = ((32, 1, 1), (64, 2, 2), (128, 2, 2), (128, 2, 2))
    PYRAMID = 48

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([127.5, 127.5, 127.5]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([64.0, 64.0, 64.0]).view(1, 3, 1, 1))
        self.stem = nn.Sequential(_conv_bn_relu(3, 16, 3, 2), _conv_bn_relu(16, 32, 3, 2))
        channels = 32
        self.stages = nn.ModuleList()
        for stage_channels, blocks, stride in self.STAGES:
            layers = []
            for block in range(blocks):
                layers.append(BasicBlock(channels, stage_channels, stride if block == 0 else 1))
                channels = stage_channels
            self.stages.append(nn.Sequential(*layers))
        self.laterals = nn.ModuleList(nn.Conv2d(stage[0], self.PYRAMID, 1) for stage in self.STAGES)
        self.smooth = _conv_bn_relu(self.PYRAMID, self.PYRAMID, 3)
        self.heatmaps = nn.Conv2d(self.PYRAMID, 4, 1)
        self.present = nn.Linear(channels, 1)
        centres = (torch.arange(GRID, dtype=torch.float32) + 0.5) / GRID
        self.register_buffer("centres", centres)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem((image - self.mean) / self.std)
        levels = []
        for stage in self.stages:
            x = stage(x)
            levels.append(x)
        present = self.present(x.mean(dim=(2, 3))).squeeze(1)
        merged = self.laterals[-1](levels[-1])
        for lateral, level in zip(reversed(self.laterals[:-1]), reversed(levels[:-1])):
            merged = lateral(level) + F.interpolate(merged, scale_factor=2.0, mode="nearest")
        logits = self.heatmaps(self.smooth(merged))
        return corners_from(logits, self.centres), present, logits


def corners_from(logits: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
    """The mean position under each heatmap, N x 4 x 2 as (x, y) fractions."""
    weights = torch.softmax(logits.flatten(2), dim=2).view(-1, 4, GRID, GRID)
    return torch.stack([(weights.sum(dim=2) * centres).sum(dim=2), (weights.sum(dim=3) * centres).sum(dim=2)], dim=2)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
