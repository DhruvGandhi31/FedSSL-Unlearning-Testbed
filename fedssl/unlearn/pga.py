"""Projected gradient ascent (Halimi et al. 2022): ascend the forget client's loss inside an
L2 ball around a reference model, then a few recovery FedAvg rounds without that client."""
from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from fedssl.config import Config
from fedssl.data import ClientData, DeviceData
from fedssl.metrics import MetricsWriter, accuracy_on, evaluate
from fedssl.models import WideResNet
from fedssl.server import GLOBAL, fedavg, without_client
from fedssl.ssl import weak_augment


def reference_weights(final: dict[str, torch.Tensor], last_update: dict[str, torch.Tensor],
                      n_participants: int) -> dict[str, torch.Tensor]:
    """Final global minus the forget client's share of its last FedAvg round. Approximate:
    its last local model is taken as final + last_update."""
    return {k: v - last_update[k].to(v.device, v.dtype) / (n_participants - 1) if k in last_update else v
            for k, v in final.items()}


@torch.no_grad()
def project_(params: list[torch.Tensor], ref: list[torch.Tensor], radius: float) -> float:
    """In place: pull params back into the L2 ball of `radius` around ref. Returns the distance before."""
    dist = torch.sqrt(sum(((p - r) ** 2).sum() for p, r in zip(params, ref))).item()
    if dist > radius:
        for p, r in zip(params, ref):
            p.copy_(r + (p - r) * (radius / dist))
    return dist


def pga(cfg: Config, data: DeviceData, clients: list[ClientData], schedule: list[list[int]],
        final_weights: dict[str, torch.Tensor], last_updates: dict[int, dict[str, torch.Tensor]],
        model: nn.Module, out_dir: Path, log: MetricsWriter,
        labeler: nn.Module | None | str = GLOBAL) -> dict[str, torch.Tensor]:
    """labeler applies to the recovery rounds, which are where SSL can re-inject knowledge."""
    u = cfg.unlearn
    device = data.x_train.device
    rounds_in = [r for r, s in enumerate(schedule) if u.forget_client in s]
    if not rounds_in:
        raise ValueError(f"client {u.forget_client} never participated; nothing to unlearn")
    model.load_state_dict(reference_weights(final_weights, last_updates[u.forget_client],
                                            len(schedule[rounds_in[-1]])))
    ref = [p.detach().clone() for p in model.parameters()]
    dists = []
    for _ in range(10):
        rand = WideResNet(bn_momentum=cfg.train.bn_momentum).to(device)
        dists.append(torch.sqrt(sum(((p - r) ** 2).sum() for p, r in zip(rand.parameters(), ref))).item())
    radius = u.pga_radius_frac * sum(dists) / len(dists)

    # eval mode: BN uses (and keeps) the reference's running stats while ascending
    model.eval()
    opt = torch.optim.SGD(model.parameters(), lr=u.pga_lr)
    lab = torch.as_tensor(clients[u.forget_client].labeled, device=device)
    acc = accuracy_on(model, data, lab)
    log.write({"pga_step": 0, "radius": radius, "forget_client_acc": acc, "eval": evaluate(model, data)})
    for step in range(1, u.pga_steps + 1):
        idx = lab[torch.randint(len(lab), (cfg.train.batch_size,), device=device)]
        loss = -F.cross_entropy(model(weak_augment(data.x_train[idx].float() / 255)), data.y_train[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        dist = project_(list(model.parameters()), ref, radius)
        if step % 10 == 0 or step == u.pga_steps:
            acc = accuracy_on(model, data, lab)
            log.write({"pga_step": step, "loss": -loss.item(), "dist": dist, "forget_client_acc": acc})
            if acc <= u.pga_stop_acc:
                break
    ev = evaluate(model, data)
    log.write({"pga_step": step, "forget_client_acc": acc, "eval": ev})
    k = cfg.data.forget_class
    print(f"ascent stopped at step {step}: forget-client acc {acc:.3f}  test_acc {ev['test_acc']:.4f}  "
          f"class{k}_acc {ev['class_acc'][k]:.4f}", flush=True)

    rec = copy.deepcopy(cfg)
    rec.fed.lr_schedule, rec.train.lr, rec.fed.eval_interval = "constant", u.pga_recovery_lr, 1
    rec_schedule = without_client(schedule, u.forget_client)[-u.pga_recovery_rounds:]
    return fedavg(rec, data, clients, rec_schedule, model, out_dir, log, labeler=labeler, save_history=False)
