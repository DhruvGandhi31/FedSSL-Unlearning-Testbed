"""Gold standard: retrain from the same init on the same schedule, forget client removed."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from fedssl.config import Config
from fedssl.data import ClientData, DeviceData
from fedssl.metrics import MetricsWriter
from fedssl.server import fedavg, without_client


def retrain(cfg: Config, data: DeviceData, clients: list[ClientData], schedule: list[list[int]],
            init_weights: dict[str, torch.Tensor], model: nn.Module, out_dir: Path,
            log: MetricsWriter) -> dict[str, torch.Tensor]:
    model.load_state_dict(init_weights)
    return fedavg(cfg, data, clients, without_client(schedule, cfg.unlearn.forget_client),
                  model, out_dir, log, save_history=False)
