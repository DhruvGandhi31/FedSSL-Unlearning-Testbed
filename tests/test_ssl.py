import copy

import numpy as np
import torch

from fedssl.client import local_train
from fedssl.config import TrainConfig
from fedssl.data import ClientData, DeviceData
from fedssl.models import WideResNet
from fedssl.ssl import fixmatch_loss, pseudo_labels, strong_augment, weak_augment


def test_fixmatch_mask_shape_and_threshold():
    probs = torch.tensor([[0.97, 0.03], [0.6, 0.4], [0.01, 0.99]])
    pseudo, mask = pseudo_labels(probs, threshold=0.95)
    assert mask.shape == (3,) and mask.dtype == torch.bool
    assert mask.tolist() == [True, False, True]
    assert pseudo.tolist() == [0, 0, 1]
    # masked-out samples contribute nothing
    logits = torch.randn(3, 2)
    assert torch.isclose(fixmatch_loss(logits, pseudo, torch.zeros(3, dtype=torch.bool)), torch.tensor(0.0))


def test_augment_shape_and_range():
    x = torch.rand(16, 3, 32, 32)
    for aug in (weak_augment, strong_augment):
        y = aug(x)
        assert y.shape == x.shape
        assert y.min() >= 0 and y.max() <= 1


def _tiny_setup():
    torch.manual_seed(0)
    n = 40
    data = DeviceData(torch.randint(0, 256, (n, 3, 32, 32), dtype=torch.uint8),
                      torch.arange(n) % 10, torch.zeros(1, 3, 32, 32, dtype=torch.uint8), torch.zeros(1))
    client = ClientData(0, np.arange(10), np.arange(10, n))
    cfg = TrainConfig(local_steps=2, batch_size=4, mu=2, threshold=0.0, amp=False)
    return data, client, cfg


def test_local_train_leaves_pseudo_labeler_untouched():
    data, client, cfg = _tiny_setup()
    model = WideResNet()
    labeler = WideResNet()
    before = copy.deepcopy(labeler.state_dict())
    update, stats = local_train(model.state_dict(), client, cfg, labeler, model=model, data=data, lr=0.03)
    assert all(torch.equal(before[k], v) for k, v in labeler.state_dict().items())
    assert set(update) == {k for k, v in model.state_dict().items() if v.is_floating_point()}
    assert stats["mask_rate"] == 1.0  # threshold 0: every pseudo-label is confident
    assert sum(stats["pl_count"]) == stats["n_unlabeled_seen"]


def test_local_train_supervised_modes():
    data, client, cfg = _tiny_setup()
    model = WideResNet()
    g = copy.deepcopy(model.state_dict())
    _, stats = local_train(g, client, cfg, None, model=model, data=data, lr=0.03)
    assert stats["loss_u"] == 0 and stats["n_unlabeled_seen"] == 0
    # lambda_u = 0 still logs pseudo-labels (the supervised control's propagation diagnostic)
    cfg.lambda_u = 0.0
    _, stats = local_train(g, client, cfg, WideResNet(), model=model, data=data, lr=0.03)
    assert stats["loss_u"] == 0 and stats["n_unlabeled_seen"] > 0
