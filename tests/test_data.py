import numpy as np
import pytest
import yaml

from fedssl.config import DataConfig, load_config
from fedssl.data import class_counts, make_clients

# Synthetic CIFAR-shaped labels: 10 classes x 500, no download needed.
LABELS = np.repeat(np.arange(10), 500)


def _all_indices(clients):
    return np.concatenate([np.concatenate([c.labeled, c.unlabeled]) for c in clients])


@pytest.mark.parametrize("scenario", ["none", "exclusive_class"])
def test_partition_covers_every_index_once(scenario):
    clients = make_clients(LABELS, DataConfig(scenario=scenario))
    idx = _all_indices(clients)
    assert len(idx) == len(LABELS)
    assert np.array_equal(np.sort(idx), np.arange(len(LABELS)))
    for c in clients:
        assert len(np.intersect1d(c.labeled, c.unlabeled)) == 0


def test_partition_depends_only_on_data_seed():
    a = make_clients(LABELS, DataConfig(data_seed=1))
    b = make_clients(LABELS, DataConfig(data_seed=1))
    c = make_clients(LABELS, DataConfig(data_seed=2))
    assert all(np.array_equal(x.labeled, y.labeled) and np.array_equal(x.unlabeled, y.unlabeled)
               for x, y in zip(a, b))
    assert not all(np.array_equal(x.labeled, y.labeled) for x, y in zip(a, c))


def test_labeled_fraction():
    cfg = DataConfig(scenario="none")
    clients = make_clients(LABELS, cfg)
    n_lab = sum(len(c.labeled) for c in clients)
    assert abs(n_lab / len(LABELS) - cfg.labeled_frac) < 0.02


def test_exclusive_class_labels_only_on_client_0():
    cfg = DataConfig(scenario="exclusive_class", forget_class=3, forget_share=0.2)
    clients = make_clients(LABELS, cfg)
    k = cfg.forget_class
    assert class_counts(clients[0].labeled, LABELS)[k] > 0
    for c in clients[1:]:
        assert class_counts(c.labeled, LABELS)[k] == 0
    # class k still reaches other clients, as unlabeled data
    assert sum(class_counts(c.unlabeled, LABELS)[k] for c in clients[1:]) > 0
    k0 = class_counts(np.concatenate([clients[0].labeled, clients[0].unlabeled]), LABELS)[k]
    assert k0 == round(cfg.forget_share * 500)


def test_subset():
    clients = make_clients(LABELS, DataConfig(subset=2000, num_clients=3, min_client_size=10))
    assert len(_all_indices(clients)) == 2000


def test_unknown_config_key_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"data": {"alpah": 0.3}}))
    with pytest.raises(KeyError):
        load_config(p)


def test_string_float_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("data:\n  alpha: 3e-1\n")  # PyYAML reads this as a string
    with pytest.raises(TypeError):
        load_config(p)
