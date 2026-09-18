import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Soft Euler characteristic
# ---------------------------------------------------------------------
def soft_euler(mask):
   
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(
            "mask must have shape [B, 1, H, W], "
            f"got {tuple(mask.shape)}"
        )

    p = mask[:, 0].clamp(
        min=0.0,
        max=1.0,
    )

    # -------------------------------------------------------------
    # 8-neighbourhood soft adjacency
    # -------------------------------------------------------------
    # Horizontal / vertical neighbours
    right = p[:, :, :-1] * p[:, :, 1:]

    down = p[:, :-1, :] * p[:, 1:, :]

    # Diagonal neighbours
    down_right = (
        p[:, :-1, :-1]
        * p[:, 1:, 1:]
    )

    down_left = (
        p[:, :-1, 1:]
        * p[:, 1:, :-1]
    )

    # -------------------------------------------------------------
    # Soft vertex contribution
    # -------------------------------------------------------------
    vertices = p.sum(
        dim=(1, 2)
    )

    # -------------------------------------------------------------
    # Soft edge contribution
    #
    # Each undirected neighbour relation is counted once.
    # -------------------------------------------------------------
    edges = (
        right.sum(dim=(1, 2))
        + down.sum(dim=(1, 2))
        + down_right.sum(dim=(1, 2))
        + down_left.sum(dim=(1, 2))
    )

    # -------------------------------------------------------------
    # Soft 2x2 face contribution
    #
    # A face is activated when all four pixels in a 2x2
    # neighbourhood have foreground probability.
    # -------------------------------------------------------------
    faces = (
        p[:, :-1, :-1]
        * p[:, :-1, 1:]
        * p[:, 1:, :-1]
        * p[:, 1:, 1:]
    )

    face_count = faces.sum(
        dim=(1, 2)
    )

    # -------------------------------------------------------------
    # Euler characteristic
    #
    # χ = V - E + F
    # -------------------------------------------------------------
    return (
        vertices
        - edges
        + face_count
    )

# ---------------------------------------------------------------------
# Topology loss
# ---------------------------------------------------------------------
def topology_loss(
    prob,
    target,
    valid_classes=None,
):
   
    num_classes = prob.shape[1]

    if valid_classes is None:

        valid_classes = torch.ones(
            num_classes,
            dtype=torch.bool,
            device=prob.device,
        )

    elif not torch.is_tensor(
        valid_classes
    ):

        valid_classes = torch.as_tensor(
            valid_classes,
            dtype=torch.bool,
            device=prob.device,
        )

    valid_classes = valid_classes.to(
        device=prob.device,
        dtype=torch.bool,
    )

    losses = []

    for class_id in range(
        1,
        num_classes,
    ):

        if not valid_classes[class_id]:
            continue

        predicted_mask = (
            prob[:, class_id:class_id + 1]
        )

        target_mask = (
            (target == class_id)
            .float()
            .unsqueeze(1)
        )

        predicted_euler = soft_euler(
            predicted_mask
        )

        target_euler = soft_euler(
            target_mask
        )

        loss = (
            predicted_euler
            - target_euler
        ).abs().mean()

        losses.append(loss)

    if not losses:
        return prob.new_tensor(0.0)

    return torch.stack(
        losses
    ).mean()


# ---------------------------------------------------------------------
# Local soft adjacency
# ---------------------------------------------------------------------
def local_soft_adjacency(probability):
    
    if probability.ndim != 4:
        raise ValueError(
            "probability must have shape "
            "[B, 1, H, W]."
        )

    if probability.shape[1] != 1:
        raise ValueError(
            "probability must contain one class."
        )

    p = probability.clamp(
        min=0.0,
        max=1.0,
    )

    neighbours = []

    # -------------------------------------------------------------
    # 8-neighbourhood
    # -------------------------------------------------------------
    neighbours.append(
        p[:, :, :, :-1]
        * p[:, :, :, 1:]
    )

    neighbours.append(
        p[:, :, :-1, :]
        * p[:, :, 1:, :]
    )

    neighbours.append(
        p[:, :, :-1, :-1]
        * p[:, :, 1:, 1:]
    )

    neighbours.append(
        p[:, :, :-1, 1:]
        * p[:, :, 1:, :-1]
    )

    # -------------------------------------------------------------
    # Mean local adjacency
    # -------------------------------------------------------------
    values = []

    for adjacency in neighbours:

        values.append(
            adjacency.mean(
                dim=(1, 2, 3)
            )
        )

    return torch.stack(
        values,
        dim=1,
    ).mean(dim=1)


# ---------------------------------------------------------------------
# Hard local adjacency
# ---------------------------------------------------------------------
def hard_local_adjacency(mask):
   
    if mask.ndim != 4:
        raise ValueError(
            "mask must have shape [B, 1, H, W]."
        )

    if mask.shape[1] != 1:
        raise ValueError(
            "mask must contain one class."
        )

    p = mask.float().clamp(
        min=0.0,
        max=1.0,
    )

    neighbours = []

    neighbours.append(
        p[:, :, :, :-1]
        * p[:, :, :, 1:]
    )

    neighbours.append(
        p[:, :, :-1, :]
        * p[:, :, 1:, :]
    )

    neighbours.append(
        p[:, :, :-1, :-1]
        * p[:, :, 1:, 1:]
    )

    neighbours.append(
        p[:, :, :-1, 1:]
        * p[:, :, 1:, :-1]
    )

    values = []

    for adjacency in neighbours:

        values.append(
            adjacency.mean(
                dim=(1, 2, 3)
            )
        )

    return torch.stack(
        values,
        dim=1,
    ).mean(dim=1)


# ---------------------------------------------------------------------
# Connectivity loss
# ---------------------------------------------------------------------
def connectivity_loss(
    prob,
    target=None,
    valid_classes=None,
):
    
    num_classes = prob.shape[1]

    if valid_classes is None:

        valid_classes = torch.ones(
            num_classes,
            dtype=torch.bool,
            device=prob.device,
        )

    elif not torch.is_tensor(
        valid_classes
    ):

        valid_classes = torch.as_tensor(
            valid_classes,
            dtype=torch.bool,
            device=prob.device,
        )

    valid_classes = valid_classes.to(
        device=prob.device,
        dtype=torch.bool,
    )

    losses = []

    for class_id in range(
        1,
        num_classes,
    ):

        if not valid_classes[class_id]:
            continue

        predicted = (
            prob[:, class_id:class_id + 1]
        )

        predicted_adjacency = (
            local_soft_adjacency(
                predicted
            )
        )

        if target is not None:

            target_class = (
                (target == class_id)
                .float()
                .unsqueeze(1)
            )

            target_adjacency = (
                hard_local_adjacency(
                    target_class
                )
            )

            class_loss = (
                predicted_adjacency
                - target_adjacency
            ).abs().mean()

        else:
                class_loss = (
                1.0
                - predicted_adjacency
            ).mean()

        losses.append(
            class_loss
        )

    if not losses:
        return prob.new_tensor(0.0)

    return torch.stack(
        losses
    ).mean()


# ---------------------------------------------------------------------
# Public alias used by the model
# ---------------------------------------------------------------------
def soft_euler_characteristic(mask):
    """
    Public name used by KE_NTMN.
    """

    return soft_euler(mask)
