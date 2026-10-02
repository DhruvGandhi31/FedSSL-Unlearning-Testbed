import torch

from fedssl.unlearn.federaser import calibrate
from fedssl.unlearn.pga import project_, reference_weights


def test_calibrate_keeps_old_norm_and_new_direction():
    old = {"w": torch.tensor([3.0, 4.0])}  # norm 5
    new = {"w": torch.tensor([0.0, 0.5])}
    out = calibrate(old, new)["w"]
    assert torch.isclose(out.norm(), torch.tensor(5.0))
    assert torch.allclose(out, torch.tensor([0.0, 5.0]))


def test_project_into_ball():
    ref = [torch.zeros(3), torch.zeros(2)]
    params = [torch.full((3,), 2.0), torch.full((2,), 2.0)]  # distance sqrt(20)
    project_(params, ref, radius=1.0)
    dist = torch.sqrt(sum(((p - r) ** 2).sum() for p, r in zip(params, ref)))
    assert torch.isclose(dist, torch.tensor(1.0))
    inside = [torch.full((3,), 0.1)]
    project_(inside, [torch.zeros(3)], radius=1.0)
    assert torch.allclose(inside[0], torch.full((3,), 0.1))  # untouched


def test_reference_removes_forget_client_share():
    # 5 participants: global = mean of local models; dropping one local model of delta d
    final = {"w": torch.tensor([1.0]), "n": torch.tensor(7)}
    ref = reference_weights(final, {"w": torch.tensor([4.0])}, n_participants=5)
    assert torch.allclose(ref["w"], torch.tensor([0.0]))
    assert ref["n"] is final["n"]
