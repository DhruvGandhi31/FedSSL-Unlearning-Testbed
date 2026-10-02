"""One client's local FixMatch training, with the pseudo-labeler passed in explicitly."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from fedssl.config import TrainConfig
from fedssl.data import NUM_CLASSES, ClientData, DeviceData
from fedssl.ssl import fixmatch_loss, pseudo_labels, strong_augment, weak_augment


def local_train(global_weights: dict[str, torch.Tensor], client: ClientData, cfg: TrainConfig,
                pseudo_labeler: nn.Module | None, *, model: nn.Module, data: DeviceData,
                lr: float) -> tuple[dict[str, torch.Tensor], dict]:
    """Runs cfg.local_steps from global_weights. pseudo_labeler is a frozen model (eval mode)
    labelling weak views; None disables the unlabeled branch. Returns (weight delta, stats)."""
    device = data.x_train.device
    model.load_state_dict(global_weights)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, init_scale=2.0 ** 10, enabled=use_amp)

    lab = torch.as_tensor(client.labeled, device=device)
    unl = torch.as_tensor(client.unlabeled, device=device)
    use_unl = pseudo_labeler is not None and len(unl) > 0
    train_unl = use_unl and cfg.lambda_u > 0  # with lambda_u = 0 pseudo-labels are still logged
    if use_unl:
        pseudo_labeler.eval()

    bs, n_u = cfg.batch_size, cfg.mu * cfg.batch_size
    pl_count = torch.zeros(NUM_CLASSES, dtype=torch.long, device=device)
    pl_correct = torch.zeros_like(pl_count)
    sum_lx = torch.zeros((), device=device)
    sum_lu = torch.zeros((), device=device)

    for _ in range(cfg.local_steps):
        li = lab[torch.randint(len(lab), (bs,), device=device)]
        x_l, y_l = weak_augment(data.x_train[li].float() / 255), data.y_train[li]
        if use_unl:
            ui = unl[torch.randint(len(unl), (n_u,), device=device)]
            x_u = data.x_train[ui].float() / 255
            with torch.no_grad(), torch.autocast(device.type, enabled=use_amp):
                probs = torch.softmax(pseudo_labeler(weak_augment(x_u)).float(), dim=1)
            pseudo, mask = pseudo_labels(probs, cfg.threshold)
            y_hidden = data.y_train[ui]  # logging only
            pl_count += torch.bincount(pseudo[mask], minlength=NUM_CLASSES)
            pl_correct += torch.bincount(pseudo[mask & (pseudo == y_hidden)], minlength=NUM_CLASSES)

        x_s = strong_augment(x_u) if train_unl else None
        with torch.autocast(device.type, enabled=use_amp):
            if train_unl:
                logits = model(torch.cat([x_l, x_s]))
                logits_l = logits[:bs]
                loss_u = fixmatch_loss(logits[bs:], pseudo, mask)
            else:
                logits_l = model(x_l)
                loss_u = torch.zeros((), device=device)
            loss_x = F.cross_entropy(logits_l.float(), y_l)
            loss = loss_x + cfg.lambda_u * loss_u
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sum_lx += loss_x.detach()
        sum_lu += loss_u.detach()

    local = model.state_dict()
    update = {k: local[k] - global_weights[k] for k in local if local[k].is_floating_point()}
    n_seen = cfg.local_steps * n_u if use_unl else 0
    stats = {
        "cid": client.cid,
        "loss_x": sum_lx.item() / cfg.local_steps,
        "loss_u": sum_lu.item() / cfg.local_steps,
        "n_unlabeled_seen": n_seen,
        "mask_rate": pl_count.sum().item() / max(n_seen, 1),
        "pl_count": pl_count.tolist(),      # confident pseudo-labels per predicted class
        "pl_correct": pl_correct.tolist(),  # of those, how many match the hidden true label
    }
    return update, stats
