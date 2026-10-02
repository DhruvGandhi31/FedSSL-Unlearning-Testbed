"""CIFAR-10 loading, Dirichlet partition, labeled/unlabeled split, forget scenarios."""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torchvision

from fedssl.config import Config, DataConfig, load_config

NUM_CLASSES = 10


@dataclass
class ClientData:
    cid: int
    labeled: np.ndarray    # indices into the train set
    unlabeled: np.ndarray  # indices; their true labels are for analysis only, never training


@dataclass
class DeviceData:
    """Whole dataset resident on the training device; uint8 NCHW images."""
    x_train: torch.Tensor
    y_train: torch.Tensor  # for unlabeled indices: analysis/logging only, never a training target
    x_test: torch.Tensor
    y_test: torch.Tensor


def to_device(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
              device: torch.device) -> DeviceData:
    img = lambda x: torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(device)
    return DeviceData(img(x_train), torch.from_numpy(y_train).to(device),
                      img(x_test), torch.from_numpy(y_test).to(device))


def load_cifar10(root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (x_train, y_train, x_test, y_test); x is uint8 NHWC, y is int64."""
    train = torchvision.datasets.CIFAR10(root, train=True, download=True)
    test = torchvision.datasets.CIFAR10(root, train=False, download=True)
    return (train.data, np.asarray(train.targets, dtype=np.int64),
            test.data, np.asarray(test.targets, dtype=np.int64))


def _dirichlet_split(idx: np.ndarray, n: int, alpha: float, rng: np.random.Generator) -> list[np.ndarray]:
    idx = rng.permutation(idx)
    cuts = (np.cumsum(rng.dirichlet(np.full(n, alpha)))[:-1] * len(idx)).astype(int)
    return np.split(idx, cuts)


def dirichlet_partition(idx: np.ndarray, labels: np.ndarray, n: int, alpha: float,
                        min_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Per-class Dirichlet(alpha) split of `idx` over n clients, redrawn until all have >= min_size."""
    for _ in range(1000):
        parts: list[list[np.ndarray]] = [[] for _ in range(n)]
        for c in np.unique(labels[idx]):
            for i, chunk in enumerate(_dirichlet_split(idx[labels[idx] == c], n, alpha, rng)):
                parts[i].append(chunk)
        out = [np.sort(np.concatenate(p)) for p in parts]
        if min(len(p) for p in out) >= min_size:
            return out
    raise RuntimeError(f"no Dirichlet draw gave every client >= {min_size} examples; lower min_client_size")


def split_labeled(idx: np.ndarray, labels: np.ndarray, frac: float, rng: np.random.Generator,
                  class_frac: dict[int, float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Stratified by class: keeps round(frac * n_c) of each class labeled (class_frac overrides per class)."""
    lab = [np.empty(0, dtype=np.int64)]
    for c in np.unique(labels[idx]):
        ci = rng.permutation(idx[labels[idx] == c])
        f = (class_frac or {}).get(int(c), frac)
        lab.append(ci[: int(round(f * len(ci)))])
    labeled = np.sort(np.concatenate(lab))
    return labeled, np.setdiff1d(idx, labeled)


def make_clients(labels: np.ndarray, cfg: DataConfig) -> list[ClientData]:
    """Deterministic in cfg.data_seed only, so original and retrain runs see identical data."""
    rng = np.random.default_rng(cfg.data_seed)
    pool = np.arange(len(labels))
    if cfg.subset:
        pool = np.sort(rng.choice(len(labels), cfg.subset, replace=False))

    if cfg.scenario == "none":
        parts = dirichlet_partition(pool, labels, cfg.num_clients, cfg.alpha, cfg.min_client_size, rng)
        return [ClientData(i, *split_labeled(p, labels, cfg.labeled_frac, rng)) for i, p in enumerate(parts)]

    if cfg.scenario == "exclusive_class":
        # Client 0 gets forget_share of class k, forget_labeled_frac of it labeled.
        # The rest of class k is spread over clients 1..N-1 as unlabeled data only.
        k = cfg.forget_class
        k_frac = cfg.labeled_frac if cfg.forget_labeled_frac < 0 else cfg.forget_labeled_frac
        is_k = labels[pool] == k
        k_idx = rng.permutation(pool[is_k])
        n0 = int(round(cfg.forget_share * len(k_idx)))
        parts = dirichlet_partition(pool[~is_k], labels, cfg.num_clients, cfg.alpha, cfg.min_client_size, rng)
        k_rest = _dirichlet_split(k_idx[n0:], cfg.num_clients - 1, cfg.alpha, rng)
        clients = []
        for i, p in enumerate(parts):
            if i == 0:
                lab, unl = split_labeled(np.concatenate([p, k_idx[:n0]]), labels, cfg.labeled_frac, rng,
                                         class_frac={k: k_frac})
            else:
                lab, unl = split_labeled(p, labels, cfg.labeled_frac, rng)
                unl = np.sort(np.concatenate([unl, k_rest[i - 1]]))
            clients.append(ClientData(i, np.sort(lab), unl))
        return clients

    raise ValueError(f"unknown scenario {cfg.scenario!r}")


def load_data(cfg: DataConfig, device: torch.device) -> tuple[DeviceData, list[ClientData]]:
    arrays = load_cifar10(cfg.root)
    return to_device(*arrays, device), make_clients(arrays[1], cfg)


def class_counts(idx: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.bincount(labels[idx], minlength=NUM_CLASSES)


def describe(clients: list[ClientData], labels: np.ndarray) -> str:
    """Per-client class counts, labeled / unlabeled (unlabeled uses hidden labels)."""
    rows = [f"{'client':>6} {'kind':>5} {'total':>6}  " + " ".join(f"{c:>5}" for c in range(NUM_CLASSES))]
    for cl in clients:
        for kind, idx in (("lab", cl.labeled), ("unl", cl.unlabeled)):
            counts = class_counts(idx, labels)
            rows.append(f"{cl.cid:>6} {kind:>5} {len(idx):>6}  " + " ".join(f"{n:>5}" for n in counts))
    return "\n".join(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print the client partition for a config.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else Config()
    _, y_train, _, _ = load_cifar10(cfg.data.root)
    print(describe(make_clients(y_train, cfg.data), y_train))
