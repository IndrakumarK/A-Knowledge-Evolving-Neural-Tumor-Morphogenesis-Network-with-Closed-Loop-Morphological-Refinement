import torch
import torch.nn.functional as F


def soft_euler(mask):
   
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(
            "mask must have shape [B, 1, H, W], "
            f"got {tuple(mask.shape)}"
        )

    p = mask[:, 0]

    # Zero padding preserves the original spatial boundary.
    q = F.pad(
        p,
        (1, 1, 1, 1),
        mode="constant",
        value=0.0,
    )

    # Vertex contribution.
    vertices = q[:, :-1, :-1]

    # Horizontal and vertical edge contributions.
    horizontal_edges = (
        q[:, :-1, 1:]
        * q[:, 1:, 1:]
    )

    vertical_edges = (
        q[:, :-1, :-1]
        * q[:, 1:, :-1]
    )

    # Four-neighbour face contribution.
    faces = (
        q[:, :-1, :-1]
        * q[:, :-1, 1:]
        * q[:, 1:, :-1]
        * q[:, 1:, 1:]
    )

    return (
        vertices.sum(dim=(1, 2))
        - horizontal_edges.sum(dim=(1, 2))
        - vertical_edges.sum(dim=(1, 2))
        + faces.sum(dim=(1, 2))
    )


def topology_loss(prob, target):
    
    losses = []

    for class_id in range(1, prob.shape[1]):
        predicted_mask = prob[:, class_id:class_id + 1]

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

        losses.append(
            (
                predicted_euler
                - target_euler
            ).abs().mean()
        )

    if not losses:
        return prob.new_tensor(0.0)

    return torch.stack(losses).mean()


def connectivity_loss(prob):
    
    losses = []

    for class_id in range(1, prob.shape[1]):
        p = prob[:, class_id:class_id + 1]

        horizontal_difference = (
            p[:, :, :, 1:]
            - p[:, :, :, :-1]
        ).abs().mean()

        vertical_difference = (
            p[:, :, 1:, :]
            - p[:, :, :-1, :]
        ).abs().mean()

        losses.extend(
            [
                horizontal_difference,
                vertical_difference,
            ]
        )

    if not losses:
        return prob.new_tensor(0.0)

    return torch.stack(losses).mean()