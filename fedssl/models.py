"""WRN-28-2, the standard FixMatch backbone (~1.47M params)."""
from __future__ import annotations

import torch
from torch import nn

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)


class Block(nn.Module):
    """Pre-activation residual block, as in the FixMatch reference implementation."""

    def __init__(self, cin: int, cout: int, stride: int, bn_momentum: float, activate_before_residual: bool):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin, momentum=bn_momentum)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.act = nn.LeakyReLU(0.1)
        self.shortcut = None if cin == cout and stride == 1 else nn.Conv2d(cin, cout, 1, stride, 0, bias=False)
        self.activate_before_residual = activate_before_residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(x))
        if self.activate_before_residual:
            x = out
        out = self.conv2(self.act(self.bn2(self.conv1(out))))
        return out + (x if self.shortcut is None else self.shortcut(x))


class WideResNet(nn.Module):
    """Takes float images in [0, 1], NCHW; normalization happens inside."""

    def __init__(self, num_classes: int = 10, depth: int = 28, widen: int = 2, bn_momentum: float = 0.1):
        super().__init__()
        n = (depth - 4) // 6
        ch = [16, 16 * widen, 32 * widen, 64 * widen]
        self.register_buffer("mean", torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CIFAR10_STD).view(1, 3, 1, 1), persistent=False)
        self.conv = nn.Conv2d(3, ch[0], 3, 1, 1, bias=False)
        layers = []
        for g, stride in enumerate((1, 2, 2)):
            for b in range(n):
                cin = ch[g] if b == 0 else ch[g + 1]
                layers.append(Block(cin, ch[g + 1], stride if b == 0 else 1, bn_momentum,
                                    activate_before_residual=(g == 0 and b == 0)))
        self.blocks = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(ch[3], momentum=bn_momentum)
        self.act = nn.LeakyReLU(0.1)
        self.fc = nn.Linear(ch[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # zero logits at init: an untrained pseudo-labeler is uniform, never confidently wrong
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(self.conv((x - self.mean) / self.std))
        x = self.act(self.bn(x))
        return self.fc(torch.flatten(nn.functional.adaptive_avg_pool2d(x, 1), 1))
