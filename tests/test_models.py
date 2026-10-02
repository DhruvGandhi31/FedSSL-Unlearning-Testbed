import torch

from fedssl.models import WideResNet, recompute_bn_stats


def test_wrn28_2_shape_and_size():
    model = WideResNet(num_classes=10)
    out = model(torch.rand(4, 3, 32, 32))
    assert out.shape == (4, 10)
    n_params = sum(p.numel() for p in model.parameters())
    assert 1.4e6 < n_params < 1.6e6
    # normalization constants are not part of the federated state
    assert "mean" not in model.state_dict()


def test_recompute_bn_stats_matches_data():
    torch.manual_seed(0)
    model = WideResNet()
    images = torch.randint(100, 156, (64, 3, 32, 32), dtype=torch.uint8)
    recompute_bn_stats(model, images, batch_size=16)
    bn = model.blocks[0].bn1  # first BN: its input is just conv(normalized images)
    with torch.no_grad():
        expected = model.conv((images.float() / 255 - model.mean) / model.std).mean(dim=(0, 2, 3))
    assert torch.allclose(bn.running_mean, expected, atol=1e-4)
    assert bn.momentum == 0.1  # restored
