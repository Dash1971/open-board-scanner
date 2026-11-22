"""BoardNet: a small convolutional network that reads a rectified board in one pass.

Input: float32 RGB, N x 3 x 256 x 256, values 0..255 (normalisation is inside
the model). Outputs:

- square logits, N x 13 x 8 x 8, for the grid as it appears in the image (row
  0 is the top of the image; class order is boardgen.CLASSES);
- one logit per image for "White is at the bottom of the upright board";
- four logits per image for how many quarter turns counter-clockwise bring
  the image upright.

The network is a narrow ResNet: plain 3x3 convolutions, batch norm and ReLU
only, so it exports cleanly to ONNX and quantises to INT8. A depthwise
(MobileNet-style) version with 1.36M parameters was tried first; it trained
about 2.6 times slower on Apple's GPU backend and needed twice the memory.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _conv_bn_relu(inputs: int, outputs: int, kernel: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, kernel, stride, kernel // 2, bias=False),
        nn.BatchNorm2d(outputs),
        nn.ReLU(inplace=True),
    )


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with a skip connection."""

    def __init__(self, inputs: int, outputs: int, stride: int) -> None:
        super().__init__()
        self.first = _conv_bn_relu(inputs, outputs, 3, stride)
        self.second = nn.Sequential(nn.Conv2d(outputs, outputs, 3, 1, 1, bias=False), nn.BatchNorm2d(outputs))
        self.shortcut = None
        if stride != 1 or inputs != outputs:
            self.shortcut = nn.Sequential(nn.Conv2d(inputs, outputs, 1, stride, bias=False), nn.BatchNorm2d(outputs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x if self.shortcut is None else self.shortcut(x)
        return F.relu(self.second(self.first(x)) + skip)


class BoardNet(nn.Module):
    # (channels, blocks, stride) per stage. Two stem convolutions take the image
    # from 256 to 64; the stages then run at 64, 32, 16 and 8, one cell per square.
    STAGES = ((32, 2, 1), (64, 2, 2), (128, 2, 2), (192, 2, 2))

    def __init__(self, classes: int = 13) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor([127.5, 127.5, 127.5]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([64.0, 64.0, 64.0]).view(1, 3, 1, 1))
        channels = 32
        layers: list[nn.Module] = [_conv_bn_relu(3, 16, 3, 2), _conv_bn_relu(16, channels, 3, 2)]
        for stage_channels, blocks, stride in self.STAGES:
            for block in range(blocks):
                layers.append(BasicBlock(channels, stage_channels, stride if block == 0 else 1))
                channels = stage_channels
        self.features = nn.Sequential(*layers)
        self.squares = nn.Conv2d(channels, classes, 1)
        self.orientation = nn.Linear(channels, 1)
        self.rotation = nn.Linear(channels, 4)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features((image - self.mean) / self.std)
        pooled = features.mean(dim=(2, 3))
        return self.squares(features), self.orientation(pooled).squeeze(1), self.rotation(pooled)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
