import torch

def entropy(prob, eps=1e-8):
   
    if prob.ndim != 4:
        raise ValueError(
            "prob must have shape [B, C, H, W], "
            f"got {tuple(prob.shape)}"
        )

    num_classes = prob.shape[1]

    if num_classes < 2:
        raise ValueError(
            "prob must contain at least two classes."
        )

    p = prob.clamp_min(eps)

    entropy_map = -(
        p * torch.log(p)
    ).sum(dim=1)

    normalization = torch.log(
        torch.tensor(
            float(num_classes),
            device=prob.device,
            dtype=prob.dtype,
        )
    )

    return entropy_map / normalization


def consistency(prob, target):
   
    if prob.ndim != 4:
        raise ValueError(
            "prob must have shape [B, C, H, W]."
        )

    if target.ndim != 3:
        raise ValueError(
            "target must have shape [B, H, W]."
        )

    prediction = prob.argmax(dim=1)

    error = (
        prediction != target
    ).float()

    uncertainty = entropy(prob)

    return (
        uncertainty - error
    ).abs().mean()