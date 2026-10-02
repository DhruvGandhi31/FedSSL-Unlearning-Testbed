import torch

from fedssl.server import apply_update, make_schedule, without_client


def test_schedule_deterministic_per_seed():
    a = make_schedule(10, 50, 0.5, seed=0)
    assert a == make_schedule(10, 50, 0.5, seed=0)
    assert a != make_schedule(10, 50, 0.5, seed=1)
    assert len(a) == 50 and all(len(s) == 5 and len(set(s)) == 5 for s in a)


def test_retrain_schedule_only_drops_forgotten_client():
    s = make_schedule(10, 50, 0.5, seed=0)
    r = without_client(s, 0)
    for orig, re in zip(s, r):
        assert 0 not in re
        assert re == [c for c in orig if c != 0]


def test_apply_update_averages_float_keys_only():
    w = {"w": torch.zeros(2), "n": torch.tensor(3)}
    new = apply_update(w, [{"w": torch.tensor([1.0, 2.0])}, {"w": torch.tensor([3.0, 4.0])}])
    assert torch.equal(new["w"], torch.tensor([2.0, 3.0]))
    assert new["n"] is w["n"]
