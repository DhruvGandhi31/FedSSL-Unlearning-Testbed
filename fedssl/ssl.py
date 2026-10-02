"""Batched on-device weak/strong augmentations and the FixMatch loss.

Images are float NCHW in [0, 1]. Each sample draws its own random op and magnitude,
so augmenting a batch matches per-image augmentation without a CPU dataloader.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as TF

GRAY = 0.5  # fill value for geometric ops and cutout


def weak_augment(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Random horizontal flip + random translate up to `pad` px (reflect padding)."""
    n, c, h, w = x.shape
    dev = x.device
    flip = torch.rand(n, device=dev) < 0.5
    x = torch.where(flip.view(n, 1, 1, 1), x.flip(3), x)
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    rows = torch.randint(0, 2 * pad + 1, (n, 1), device=dev) + torch.arange(h, device=dev)
    cols = torch.randint(0, 2 * pad + 1, (n, 1), device=dev) + torch.arange(w, device=dev)
    return xp[torch.arange(n, device=dev).view(n, 1, 1, 1), torch.arange(c, device=dev).view(1, c, 1, 1),
              rows.view(n, 1, h, 1), cols.view(n, 1, 1, w)]


def _blend(base: torch.Tensor, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    return (base + f * (x - base)).clamp(0, 1)


def _gray(x: torch.Tensor) -> torch.Tensor:
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]).expand_as(x)


def _affine(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x - GRAY, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + GRAY


def _theta(n: int, dev: torch.device, a=0.0, b=0.0, c=0.0, d=0.0, tx=0.0, ty=0.0) -> torch.Tensor:
    """Per-sample 2x3 affine = identity + the given per-sample offsets."""
    t = torch.zeros(n, 2, 3, device=dev)
    t[:, 0, 0], t[:, 0, 1], t[:, 0, 2] = 1 + a, b, tx
    t[:, 1, 0], t[:, 1, 1], t[:, 1, 2] = c, 1 + d, ty
    return t


def _autocontrast(x: torch.Tensor) -> torch.Tensor:
    lo, hi = x.amin(dim=(2, 3), keepdim=True), x.amax(dim=(2, 3), keepdim=True)
    return torch.where(hi > lo, (x - lo) / (hi - lo).clamp_min(1e-6), x)


def _sharpen_base(x: torch.Tensor) -> torch.Tensor:
    """PIL's SMOOTH filter; border pixels are left unfiltered, as in PIL ImageEnhance.Sharpness."""
    k = torch.tensor([[1., 1., 1.], [1., 5., 1.], [1., 1., 1.]], device=x.device) / 13
    blurred = F.conv2d(x, k.view(1, 1, 3, 3).repeat(3, 1, 1, 1), padding=1, groups=3)
    out = x.clone()
    out[:, :, 1:-1, 1:-1] = blurred[:, :, 1:-1, 1:-1]
    return out


def _ops(x: torch.Tensor, m: torch.Tensor) -> list[torch.Tensor]:
    """All 14 FixMatch RandAugment ops applied to the whole batch; m in [0, 1] per sample.
    Ranges follow the FixMatch paper (Sohn et al. 2020, Table 12)."""
    n, dev = x.shape[0], x.device
    s = 2 * m - 1                    # signed magnitude in [-1, 1]
    f = (0.05 + 0.9 * m).view(n, 1, 1, 1)  # enhance factor in [0.05, 0.95]
    ang = s * math.radians(30)
    bits = 4 + torch.round(4 * m)
    q = (2 ** (8 - bits)).view(n, 1, 1, 1)
    u8 = (x * 255).round().to(torch.uint8)
    return [
        x,                                                                   # identity
        _autocontrast(x),
        TF.equalize(u8).float() / 255,
        _affine(x, _theta(n, dev, a=torch.cos(ang) - 1, b=-torch.sin(ang), c=torch.sin(ang), d=torch.cos(ang) - 1)),
        torch.where(x >= m.view(n, 1, 1, 1), 1 - x, x),                      # solarize, threshold in [0, 1]
        _blend(_gray(x), x, f),                                              # color
        torch.floor(u8.float() / q) * q / 255,                               # posterize, 4..8 bits
        _blend(_gray(x).mean(dim=(1, 2, 3), keepdim=True), x, f),            # contrast
        _blend(torch.zeros_like(x), x, f),                                   # brightness
        _blend(_sharpen_base(x), x, f),                                      # sharpness
        _affine(x, _theta(n, dev, b=0.3 * s)),                               # shear x
        _affine(x, _theta(n, dev, c=0.3 * s)),                               # shear y
        _affine(x, _theta(n, dev, tx=0.6 * s)),                              # translate x, 30% of width
        _affine(x, _theta(n, dev, ty=0.6 * s)),                              # translate y
    ]


def cutout(x: torch.Tensor, size: int = 16) -> torch.Tensor:
    n, _, h, w = x.shape
    dev = x.device
    cy = torch.randint(0, h, (n, 1, 1), device=dev)
    cx = torch.randint(0, w, (n, 1, 1), device=dev)
    ys, xs = torch.arange(h, device=dev).view(1, h, 1), torch.arange(w, device=dev).view(1, 1, w)
    hole = ((ys - cy).abs() < size // 2) & ((xs - cx).abs() < size // 2)
    return torch.where(hole.unsqueeze(1), GRAY, x)


def strong_augment(x: torch.Tensor, n_ops: int = 2) -> torch.Tensor:
    """Weak aug, then RandAugment (n_ops random ops, random magnitude), then 16px cutout."""
    x = weak_augment(x)
    n = x.shape[0]
    for _ in range(n_ops):
        op = torch.randint(0, 14, (n,), device=x.device).view(n, 1, 1, 1)
        out = x
        for i, y in enumerate(_ops(x, torch.rand(n, device=x.device))):
            out = torch.where(op == i, y, out)
        x = out
    return cutout(x)


def pseudo_labels(probs: torch.Tensor, threshold: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (hard labels, confidence mask), both shape (N,)."""
    conf, pseudo = probs.max(dim=1)
    return pseudo, conf >= threshold


def fixmatch_loss(logits_strong: torch.Tensor, pseudo: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked CE, averaged over the full unlabeled batch (not just confident samples)."""
    return (F.cross_entropy(logits_strong.float(), pseudo, reduction="none") * mask).mean()
