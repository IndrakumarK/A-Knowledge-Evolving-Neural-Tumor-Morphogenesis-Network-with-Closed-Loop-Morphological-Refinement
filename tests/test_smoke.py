import torch

from src.ketmn.model import KE_NTMN
from src.ketmn.morphology import morphology


def test_forward():
    model = KE_NTMN()

    x = torch.randn(2, 4, 64, 64)
    target = torch.randint(0, 4, (2, 64, 64))

    output = model(x, target)

    assert output["prob"].shape == (2, 4, 64, 64)
    assert output["logits"].shape == (2, 4, 64, 64)
    assert output["brain_mask"].shape == (2, 1, 64, 64)
    assert len(output["stage_probs"]) == 3
    assert len(output["states"]) == 3


def test_morphology():
    mask = torch.sigmoid(
        torch.randn(2, 1, 32, 32)
    )

    features = morphology(mask)

    assert features.shape == (2, 9)
    assert torch.isfinite(features).all()