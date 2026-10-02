"""FedAvg loop, precomputed client sampling schedule, and update history store."""
from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fedssl.client import local_train
from fedssl.config import Config
from fedssl.data import ClientData, DeviceData
from fedssl.metrics import MetricsWriter, evaluate, pseudo_label_scan


def make_schedule(num_clients: int, rounds: int, participation: float, seed: int) -> list[list[int]]:
    """Client set per round. Generated once per seed and saved; never resampled."""
    rng = np.random.default_rng(seed)
    k = max(1, round(participation * num_clients))
    return [sorted(rng.choice(num_clients, k, replace=False).tolist()) for _ in range(rounds)]


def without_client(schedule: list[list[int]], cid: int) -> list[list[int]]:
    """The retrain schedule: same rounds, `cid` dropped from each round's set."""
    return [[c for c in s if c != cid] for s in schedule]


def round_lr(cfg: Config, r: int) -> float:
    if cfg.fed.lr_schedule == "constant":
        return cfg.train.lr
    if cfg.fed.lr_schedule == "cosine":
        return cfg.train.lr * math.cos(7 * math.pi * r / (16 * cfg.fed.rounds))
    raise ValueError(f"unknown lr_schedule {cfg.fed.lr_schedule!r}")


def apply_update(weights: dict[str, torch.Tensor], updates: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Unweighted FedAvg (every client runs the same number of local steps)."""
    new = dict(weights)
    for k in updates[0]:
        new[k] = weights[k] + torch.stack([u[k] for u in updates]).mean(0)
    return new


def save_update(path: Path, update: dict[str, torch.Tensor]) -> None:
    torch.save({k: v.half().cpu() for k, v in update.items()}, path)


def fedavg(cfg: Config, data: DeviceData, clients: list[ClientData], schedule: list[list[int]],
           model: nn.Module, run_dir: Path, log: MetricsWriter) -> dict[str, torch.Tensor]:
    """Runs len(schedule) rounds from model's current weights. Writes one metrics line per round,
    fp16 client updates to run_dir/history every history_interval rounds, final.pt at the end."""
    history = run_dir / "history"
    history.mkdir(exist_ok=True)
    weights = copy.deepcopy(model.state_dict())
    labeler = copy.deepcopy(model)  # default pseudo-labeler: the current global model
    last_update: dict[int, dict[str, torch.Tensor]] = {}

    for r, participants in enumerate(schedule):
        t0 = time.time()
        lr = round_lr(cfg, r)
        labeler.load_state_dict(weights)
        updates, stats = [], []
        for cid in participants:
            u, s = local_train(weights, clients[cid], cfg.train, labeler, model=model, data=data, lr=lr)
            updates.append(u)
            stats.append(s)
            last_update[cid] = u
            if r % cfg.fed.history_interval == 0:
                save_update(history / f"r{r:04d}_c{cid:02d}.pt", u)
        if updates:
            weights = apply_update(weights, updates)

        line = {"round": r, "clients": participants, "lr": lr, "train": stats}
        if (r + 1) % cfg.fed.eval_interval == 0 or r == len(schedule) - 1:
            model.load_state_dict(weights)
            line["eval"] = evaluate(model, data)
            line["eval"]["pl_scan"] = pseudo_label_scan(model, data, clients, cfg.train.threshold)
        line["time"] = time.time() - t0
        log.write(line)
        if "eval" in line:
            k = cfg.data.forget_class
            print(f"round {r:4d}  test_acc {line['eval']['test_acc']:.4f}  "
                  f"class{k}_acc {line['eval']['class_acc'][k]:.4f}  {line['time']:.1f}s/round", flush=True)

    model.load_state_dict(weights)
    torch.save(weights, run_dir / "final.pt")
    # client 0's last local update is what PGA needs to build its reference model
    torch.save({cid: {k: v.half().cpu() for k, v in u.items()} for cid, u in last_update.items()},
               run_dir / "last_updates.pt")
    return weights
