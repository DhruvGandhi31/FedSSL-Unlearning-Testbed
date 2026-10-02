"""FedEraser (Liu et al. 2021): rebuild the model from stored updates, calibrated by short
retraining on the remaining clients."""
from __future__ import annotations

import copy
import dataclasses
import time
from pathlib import Path

import torch
from torch import nn

from fedssl.client import local_train
from fedssl.config import Config
from fedssl.data import ClientData, DeviceData
from fedssl.metrics import MetricsWriter, evaluate, pseudo_label_scan
from fedssl.server import GLOBAL, apply_update, round_lr


def calibrate(old: dict[str, torch.Tensor], new: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Per tensor: magnitude of the stored update, direction of the calibration update."""
    return {k: new[k] * (old[k].norm() / new[k].norm().clamp_min(1e-12)) for k in new}


def stored_rounds(history: Path) -> list[int]:
    return sorted({int(p.name[1:5]) for p in history.glob("r*_c*.pt")})


def federaser(cfg: Config, data: DeviceData, clients: list[ClientData], schedule: list[list[int]],
              init_weights: dict[str, torch.Tensor], history: Path, model: nn.Module, log: MetricsWriter,
              labeler: nn.Module | None | str = GLOBAL) -> dict[str, torch.Tensor]:
    """labeler for calibration: GLOBAL (the model being rebuilt), a fixed frozen model, or None."""
    device = data.x_train.device
    forget = cfg.unlearn.forget_client
    calib = dataclasses.replace(cfg.train, local_steps=max(1, round(cfg.train.local_steps * cfg.unlearn.federaser_calib_ratio)))
    weights = {k: v.to(device) for k, v in init_weights.items()}
    rebuilt = copy.deepcopy(model)

    for j, r in enumerate(stored_rounds(history)):
        t0 = time.time()
        others = [c for c in schedule[r] if c != forget]
        old = [{k: v.to(device).float() for k, v in torch.load(history / f"r{r:04d}_c{c:02d}.pt").items()}
               for c in others]
        stats = []
        if j == 0:
            updates = old  # the rebuilt and original models coincide here, nothing to calibrate
        else:
            if labeler == GLOBAL:
                rebuilt.load_state_dict(weights)
            pl = rebuilt if labeler == GLOBAL else labeler
            updates = []
            for c, u_old in zip(others, old):
                u_new, s = local_train(weights, clients[c], calib, pl, model=model, data=data, lr=round_lr(cfg, r))
                updates.append(calibrate(u_old, u_new))
                stats.append(s)
        weights = apply_update(weights, updates)

        model.load_state_dict(weights)
        ev = evaluate(model, data)
        ev["pl_scan"] = pseudo_label_scan(model, data, clients, cfg.train.threshold)
        log.write({"step": j, "round": r, "clients": others, "train": stats, "eval": ev, "time": time.time() - t0})
        k = cfg.data.forget_class
        print(f"step {j:3d} (round {r:4d})  test_acc {ev['test_acc']:.4f}  class{k}_acc {ev['class_acc'][k]:.4f}",
              flush=True)

    model.load_state_dict(weights)
    return weights
