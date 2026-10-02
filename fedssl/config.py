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
class Config:
    name: str = "unnamed"
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)


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
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _build(Config, raw, "config")


def save_config(cfg: Config, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(dataclasses.asdict(cfg), f, sort_keys=False)
