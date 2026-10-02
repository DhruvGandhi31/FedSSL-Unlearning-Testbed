"""Experiment config: plain dataclasses loaded strictly from YAML."""
from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    root: str = "data"
    num_clients: int = 10
    alpha: float = 0.3                 # Dirichlet concentration over classes
    labeled_frac: float = 0.1          # per client, stratified by class
    min_client_size: int = 100         # resample the Dirichlet draw until every client has this many
    data_seed: int = 0                 # partition and labeled split depend only on this
    subset: int = 0                    # >0: random subset of the train set (smoke runs)
    scenario: str = "exclusive_class"  # "none" | "exclusive_class"
    forget_class: int = 0              # class k for exclusive_class
    forget_share: float = 0.2          # fraction of class-k train images given to client 0


@dataclass
class TrainConfig:
    local_steps: int = 50      # fixed per round, regardless of client data size
    batch_size: int = 64       # labeled batch; unlabeled batch is mu * batch_size
    mu: int = 3
    lambda_u: float = 1.0      # 0 = supervised control
    threshold: float = 0.95    # FixMatch confidence threshold tau
    lr: float = 0.03
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True
    bn_momentum: float = 0.1   # FixMatch uses 0.001 (with an EMA model); stale stats make the eval-mode labeler confidently wrong
    amp: bool = True


@dataclass
class FedConfig:
    rounds: int = 200
    participation: float = 0.5  # fraction of clients sampled per round
    lr_schedule: str = "cosine"  # "cosine" (FixMatch's 7/16-period cosine over rounds) | "constant"
    history_interval: int = 5    # store every participant's update every this many rounds (FedEraser)
    eval_interval: int = 5


@dataclass
class UnlearnConfig:
    forget_client: int = 0
    federaser_calib_ratio: float = 0.5  # calibration local steps = ratio * train.local_steps
    pga_lr: float = 0.01
    pga_steps: int = 200                # max ascent steps
    pga_stop_acc: float = 0.1           # stop once accuracy on the forget client's labeled data <= this
    pga_radius_frac: float = 0.33       # L2 ball radius = frac * mean distance(reference, random inits)
    pga_recovery_rounds: int = 5        # FedAvg rounds without the forget client after ascent
    pga_recovery_lr: float = 0.01


@dataclass
class Config:
    name: str = "unnamed"
    seed: int = 0
    out_dir: str = "results"
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    fed: FedConfig = field(default_factory=FedConfig)
    unlearn: UnlearnConfig = field(default_factory=UnlearnConfig)


def _build(cls: type, raw: dict, where: str):
    if not isinstance(raw, dict):
        raise TypeError(f"{where}: expected a mapping, got {type(raw).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - names)
    if unknown:
        raise KeyError(f"{where}: unknown config keys {unknown}")
    kwargs = {}
    for key, value in raw.items():
        typ, path = hints[key], f"{where}.{key}"
        if dataclasses.is_dataclass(typ):
            value = _build(typ, value, path)
        elif typ is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        elif typ in (int, float, str, bool) and type(value) is not typ:
            # catches PyYAML reading "1e-3" as a string
            raise TypeError(f"{path}: expected {typ.__name__}, got {value!r}")
        kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return _build(Config, raw, "config")


def save_config(cfg: Config, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)
