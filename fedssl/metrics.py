"""Evaluation, pseudo-label provenance scan, and the metrics.jsonl writer."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from fedssl.data import NUM_CLASSES, ClientData, DeviceData


@torch.no_grad()
def _predict(model: nn.Module, x: torch.Tensor, batch: int = 1000) -> torch.Tensor:
    """fp32, eval mode. Returns softmax probabilities."""
    model.eval()
    return torch.cat([torch.softmax(model(x[i:i + batch].float() / 255), dim=1) for i in range(0, len(x), batch)])


def evaluate(model: nn.Module, data: DeviceData) -> dict:
    pred = _predict(model, data.x_test).argmax(1)
    correct = (pred == data.y_test).float()
    per_class = torch.zeros(NUM_CLASSES, device=correct.device).index_add_(0, data.y_test, correct)
    per_class /= torch.bincount(data.y_test, minlength=NUM_CLASSES).clamp_min(1)
    return {"test_acc": correct.mean().item(), "class_acc": per_class.tolist()}


def accuracy_on(model: nn.Module, data: DeviceData, idx: torch.Tensor) -> float:
    """Accuracy on train-set indices (e.g. the forget client's labeled data)."""
    return (_predict(model, data.x_train[idx]).argmax(1) == data.y_train[idx]).float().mean().item()


def pseudo_label_scan(model: nn.Module, data: DeviceData, clients: list[ClientData], threshold: float) -> list[dict]:
    """Labels every client's full unlabeled pool (no augmentation). Per client: confident
    pseudo-labels per predicted class and how many match the hidden true label."""
    out = []
    for c in clients:
        idx = torch.as_tensor(c.unlabeled, device=data.x_train.device)
        conf, pseudo = _predict(model, data.x_train[idx]).max(1)
        mask = conf >= threshold
        hit = mask & (pseudo == data.y_train[idx])  # hidden labels: analysis only
        out.append({"cid": c.cid,
                    "pl_count": torch.bincount(pseudo[mask], minlength=NUM_CLASSES).tolist(),
                    "pl_correct": torch.bincount(pseudo[hit], minlength=NUM_CLASSES).tolist()})
    return out


def git_commit(repo: Path = Path(".")) -> str | None:
    """Reads HEAD from .git without invoking git. None if unavailable."""
    git = repo / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        if (git / ref).exists():
            return (git / ref).read_text().strip()
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    return None


class MetricsWriter:
    """Appends one JSON object per line; flushed every write so a crashed run keeps its log."""

    def __init__(self, path: Path):
        self.f = open(path, "a")

    def write(self, obj: dict) -> None:
        self.f.write(json.dumps(obj) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()
