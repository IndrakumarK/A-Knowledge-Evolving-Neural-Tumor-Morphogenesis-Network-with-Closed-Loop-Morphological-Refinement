import torch

def morphology(mask, eps=1e-6):
    
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(
            "mask must have shape [B, 1, H, W], "
            f"got {tuple(mask.shape)}"
        )

    batch_size, _, height, width = mask.shape

    if height < 2 or width < 2:
        raise ValueError(
            "mask spatial dimensions must be at least 2x2."
        )

    dtype = mask.dtype
    device = mask.device

    # -------------------------------------------------------------
    # Coordinate grid
    # -------------------------------------------------------------
    yy, xx = torch.meshgrid(
        torch.arange(
            height,
            device=device,
            dtype=dtype,
        ),
        torch.arange(
            width,
            device=device,
            dtype=dtype,
        ),
        indexing="ij",
    )

    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)

    m = mask[:, 0].clamp(
        min=0.0,
        max=1.0,
    )

    # -------------------------------------------------------------
    # 1. Soft area
    # -------------------------------------------------------------
    area = (
        m.sum(
            dim=(1, 2),
            keepdim=True,
        )
        + eps
    )

    # -------------------------------------------------------------
    # 2. Soft centroid
    # -------------------------------------------------------------
    cx = (
        (m * xx).sum(
            dim=(1, 2),
            keepdim=True,
        )
        / area
    )

    cy = (
        (m * yy).sum(
            dim=(1, 2),
            keepdim=True,
        )
        / area
    )

    # -------------------------------------------------------------
    # Centered coordinates
    # -------------------------------------------------------------
    dx = xx - cx
    dy = yy - cy

    # -------------------------------------------------------------
    # 3. Soft covariance
    #
    # Covariance components:
    #   Cxx
    #   Cxy
    #   Cyy
    # -------------------------------------------------------------
    cxx = (
        (m * dx.pow(2)).sum(
            dim=(1, 2),
            keepdim=True,
        )
        / area
    )

    cyy = (
        (m * dy.pow(2)).sum(
            dim=(1, 2),
            keepdim=True,
        )
        / area
    )

    cxy = (
        (m * dx * dy).sum(
            dim=(1, 2),
            keepdim=True,
        )
        / area
    )

    covariance = torch.cat(
        [
            cxx.flatten(1),
            cxy.flatten(1),
            cyy.flatten(1),
        ],
        dim=1,
    )

    # -------------------------------------------------------------
    # 4. Elongation
    # -------------------------------------------------------------
    covariance_matrix = torch.stack(
        [
            cxx.flatten(),
            cxy.flatten(),
            cxy.flatten(),
            cyy.flatten(),
        ],
        dim=1,
    ).reshape(
        batch_size,
        2,
        2,
    )

    eigenvalues = torch.linalg.eigvalsh(
        covariance_matrix
    )

    eigenvalues = eigenvalues.clamp_min(
        eps
    )

    elongation = (
        eigenvalues[:, 1]
        / eigenvalues[:, 0]
    ).sqrt().unsqueeze(1)

    # -------------------------------------------------------------
    # 5. Soft boundary
    # -------------------------------------------------------------
    horizontal_gradient = torch.zeros_like(m)
    vertical_gradient = torch.zeros_like(m)

    horizontal_gradient[:, :, :-1] = torch.abs(
        m[:, :, 1:]
        - m[:, :, :-1]
    )

    vertical_gradient[:, :-1, :] = torch.abs(
        m[:, 1:, :]
        - m[:, :-1, :]
    )

    boundary = (
        horizontal_gradient
        + vertical_gradient
    ).mean(
        dim=(1, 2)
    ).unsqueeze(1)

    # -------------------------------------------------------------
    # 6. Soft compactness
    # -------------------------------------------------------------
    perimeter_proxy = (
        boundary + eps
    )

    compactness = (
        4.0
        * torch.pi
        * area.flatten(1)
        / perimeter_proxy.pow(2)
    )

    # -------------------------------------------------------------
    # Final morphology vector
    # -------------------------------------------------------------
    descriptors = torch.cat(
        [
            area.flatten(1),
            cx.flatten(1),
            cy.flatten(1),
            covariance,
            elongation,
            boundary,
            compactness,
        ],
        dim=1,
    )

    if descriptors.shape[1] != 9:
        raise RuntimeError(
            "Expected a 9-dimensional morphology vector, "
            f"got {descriptors.shape[1]}."
        )

    return descriptors
