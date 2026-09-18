import torch
# ---------------------------------------------------------------------
# Normalized predictive entropy
# ---------------------------------------------------------------------
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

    # Prevent log(0).
    p = prob.clamp(
        min=eps,
        max=1.0,
    )

    # -------------------------------------------------------------
    # Predictive entropy
    # -------------------------------------------------------------
    entropy_map = -(
        p * torch.log(p)
    ).sum(
        dim=1
    )

    # -------------------------------------------------------------
    # Normalize entropy to approximately [0, 1].
    # Maximum entropy = log(C).
    # -------------------------------------------------------------
    normalization = torch.log(
        torch.tensor(
            float(num_classes),
            device=prob.device,
            dtype=prob.dtype,
        )
    ).clamp_min(eps)

    normalized_entropy = (
        entropy_map
        / normalization
    )

    return normalized_entropy

# ---------------------------------------------------------------------
# Normalized predictive entropy alias
# ---------------------------------------------------------------------
def normalized_predictive_entropy(
    prob,
    eps=1e-8,
):
   
    return entropy(
        prob,
        eps=eps,
    )

# ---------------------------------------------------------------------
# Uncertainty-error consistency
# ---------------------------------------------------------------------
def consistency(
    prob,
    target,
    eps=1e-8,
):

    if prob.ndim != 4:
        raise ValueError(
            "prob must have shape [B, C, H, W]."
        )

    if target.ndim != 3:
        raise ValueError(
            "target must have shape [B, H, W]."
        )

    if (
        prob.shape[0] != target.shape[0]
        or prob.shape[2] != target.shape[1]
        or prob.shape[3] != target.shape[2]
    ):
        raise ValueError(
            "Probability and target spatial dimensions "
            "do not match."
        )

    num_classes = prob.shape[1]

    # -------------------------------------------------------------
    # Ground-truth labels must be valid class indices.
    # -------------------------------------------------------------
    if target.min() < 0 or target.max() >= num_classes:
        raise ValueError(
            "Target contains class indices outside "
            f"[0, {num_classes - 1}]."
        )

    # -------------------------------------------------------------
    # Normalized predictive entropy
    # -------------------------------------------------------------
    uncertainty = entropy(
        prob,
        eps=eps,
    )

    # -------------------------------------------------------------
    # Probability assigned to the ground-truth class
    # -------------------------------------------------------------
    target_probability = (
        prob.gather(
            dim=1,
            index=target.unsqueeze(1),
        )
        .squeeze(1)
    )

    # -------------------------------------------------------------
    # Soft local prediction error
    # -------------------------------------------------------------
    error = (
        1.0
        - target_probability
    )

    # -------------------------------------------------------------
    # Uncertainty-error consistency
    # -------------------------------------------------------------
    return (
        uncertainty
        - error
    ).abs().mean()

# ---------------------------------------------------------------------
# Public uncertainty-consistency alias
# ---------------------------------------------------------------------
def uncertainty_consistency(
    prob,
    target,
    eps=1e-8,
):
    
    return consistency(
        prob,
        target,
        eps=eps,
    )
