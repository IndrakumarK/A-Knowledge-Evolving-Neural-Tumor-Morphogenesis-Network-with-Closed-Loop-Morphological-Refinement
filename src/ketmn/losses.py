import torch
import torch.nn.functional as F


def dice_loss(pred, target, num_classes=4, smooth=1e-6):
    probs = F.softmax(pred, dim=1)

    target_one_hot = F.one_hot(
        target.long(),
        num_classes=num_classes
    ).permute(0, 3, 1, 2).float()

    # Exclude background class.
    probs = probs[:, 1:]
    target_one_hot = target_one_hot[:, 1:]

    intersection = (probs * target_one_hot).sum(dim=(0, 2, 3))
    denominator = (
        probs.sum(dim=(0, 2, 3))
        + target_one_hot.sum(dim=(0, 2, 3))
    )

    dice = (2.0 * intersection + smooth) / (
        denominator + smooth
    )

    return 1.0 - dice.mean()


def smoothness_loss(pred):
    
    probs = F.softmax(pred, dim=1)

    loss_h = torch.abs(
        probs[:, :, 1:, :] - probs[:, :, :-1, :]
    ).mean()

    loss_w = torch.abs(
        probs[:, :, :, 1:] - probs[:, :, :, :-1]
    ).mean()

    return loss_h + loss_w


def structural_stability_loss(current_state, previous_state):
   
    if previous_state is None:
        return current_state.new_tensor(0.0)

    return F.mse_loss(current_state, previous_state.detach())


def morphology_estimation_loss(predicted_morphology, target_morphology):
   
    return F.mse_loss(
        predicted_morphology,
        target_morphology
    )


def knowledge_consistency_loss(
    predicted_morphology,
    expected_morphology,
):
    
    return F.mse_loss(
        predicted_morphology,
        expected_morphology
    )


def topology_loss(prediction, target):
    
    pred_prob = F.softmax(prediction, dim=1)[:, 1:].sum(dim=1)
    target_fg = (target > 0).float()

    pred_gradient_h = torch.abs(
        pred_prob[:, 1:, :] - pred_prob[:, :-1, :]
    )

    target_gradient_h = torch.abs(
        target_fg[:, 1:, :] - target_fg[:, :-1, :]
    )

    pred_gradient_w = torch.abs(
        pred_prob[:, :, 1:] - pred_prob[:, :, :-1]
    )

    target_gradient_w = torch.abs(
        target_fg[:, :, 1:] - target_fg[:, :, :-1]
    )

    return F.l1_loss(
        pred_gradient_h,
        target_gradient_h
    ) + F.l1_loss(
        pred_gradient_w,
        target_gradient_w
    )


def connectivity_loss(prediction):
  
    pred_prob = F.softmax(prediction, dim=1)[:, 1:].sum(dim=1)

    horizontal = torch.abs(
        pred_prob[:, :, 1:] - pred_prob[:, :, :-1]
    )

    vertical = torch.abs(
        pred_prob[:, 1:, :] - pred_prob[:, :-1, :]
    )

    return horizontal.mean() + vertical.mean()


def uncertainty_consistency_loss(
    prediction,
    target,
):
   
    probs = F.softmax(prediction, dim=1)

    entropy = -(
        probs * torch.log(probs.clamp_min(1e-8))
    ).sum(dim=1)

    entropy = entropy / torch.log(
        torch.tensor(
            probs.shape[1],
            device=probs.device,
            dtype=probs.dtype,
        )
    )

    valid_region = (target >= 0).float()

    return (entropy * valid_region).mean()


def mbnss_loss(
    brain_prediction,
    brain_target,
):
   
    dice = dice_loss(
        brain_prediction,
        brain_target,
        num_classes=2,
    )

    smooth = smoothness_loss(brain_prediction)

    return (
        1.00 * dice
        + 0.50 * smooth
    )


def total_loss(
    prediction,
    target,
    structural_stability=0.0,
    morphology_estimation=0.0,
    knowledge_consistency=0.0,
    topology=0.0,
    connectivity=0.0,
    smoothness=0.0,
    uncertainty_consistency=0.0,
):
   
    foreground_dice = dice_loss(
        prediction,
        target,
        num_classes=4,
    )

    return (
        1.00 * foreground_dice
        + 0.20 * structural_stability
        + 0.20 * morphology_estimation
        + 0.20 * knowledge_consistency
        + 0.10 * topology
        + 0.10 * connectivity
        + 0.05 * smoothness
        + 0.05 * uncertainty_consistency
    )