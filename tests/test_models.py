import torch

from fedssl.models import WideResNet


def test_wrn28_2_shape_and_size():
    model = WideResNet(num_classes=10)
    out = model(torch.rand(4, 3, 32, 32))
    assert out.shape == (4, 10)
    n_params = sum(p.numel() for p in model.parameters())
    assert 1.4e6 < n_params < 1.6e6
    # normalization constants are not part of the federated state
    assert "mean" not in model.state_dict()
