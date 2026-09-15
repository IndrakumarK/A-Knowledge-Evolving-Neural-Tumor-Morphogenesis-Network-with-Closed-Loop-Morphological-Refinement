import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.ketmn.data import BrainTumorDataset
from src.ketmn.losses import (
    connectivity_loss,
    knowledge_consistency_loss,
    morphology_estimation_loss,
    smoothness_loss,
    structural_stability_loss,
    topology_loss,
    uncertainty_consistency_loss,
    dice_loss,
)
from src.ketmn.model import KE_NTMN
from src.ketmn.morphology import morphology
from src.ketmn.uncertainty import entropy


def set_seed(seed):
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device, num_classes):
    """Evaluate mean patient-level foreground Dice."""

    model.eval()
    scores = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            target = batch["mask"].to(device)

            output = model(images)

            prediction = output["prob"].argmax(dim=1)

            class_scores = []

            for class_id in range(1, num_classes):
                pred_class = prediction == class_id
                target_class = target == class_id

                denominator = (
                    pred_class.sum()
                    + target_class.sum()
                )

                if denominator == 0:
                    score = torch.tensor(
                        1.0,
                        device=device,
                    )
                else:
                    intersection = (
                        pred_class & target_class
                    ).sum()

                    score = (
                        2.0 * intersection.float()
                        / denominator.float()
                    )

                class_scores.append(score)

            scores.append(
                torch.stack(class_scores).mean().item()
            )

    return float(np.mean(scores)) if scores else 0.0


def compute_loss(output, target):
    """
    Compute the manuscript-aligned composite training objective.

    The repository implementation uses the available differentiable
    structural signals returned by KE-NTMN.
    """

    prob = output["prob"]
    states = output["states"]
    morphology_pred = output["morphology"]

    # -------------------------------------------------------------
    # Foreground Dice
    # -------------------------------------------------------------
    dice = dice_loss(
        output["logits"],
        target,
        num_classes=prob.shape[1],
    )

    # -------------------------------------------------------------
    # Structural stability
    # -------------------------------------------------------------
    stability_terms = []

    for i in range(1, len(states)):
        stability_terms.append(
            structural_stability_loss(
                states[i],
                states[i - 1],
            )
        )

    if stability_terms:
        stability = torch.stack(
            stability_terms
        ).mean()
    else:
        stability = prob.new_tensor(0.0)

    # -------------------------------------------------------------
    # Morphological supervision
    # -------------------------------------------------------------
    foreground_target = (
        (target > 0)
        .float()
        .unsqueeze(1)
    )

    target_morphology = morphology(
        foreground_target
    )

    morphology_loss_value = morphology_estimation_loss(
        morphology_pred,
        target_morphology.detach(),
    )

    # -------------------------------------------------------------
    # Knowledge consistency
    # -------------------------------------------------------------
    expected_morphology = (
        output["morphology"]
        .detach()
    )

    knowledge_loss = knowledge_consistency_loss(
        morphology_pred,
        expected_morphology,
    )

    # -------------------------------------------------------------
    # Topology and connectivity
    # -------------------------------------------------------------
    topology = topology_loss(
        prob,
        target,
    )

    connectivity = connectivity_loss(
        prob,
    )

    # -------------------------------------------------------------
    # Spatial smoothness
    # -------------------------------------------------------------
    smoothness = smoothness_loss(
        output["logits"]
    )

    # -------------------------------------------------------------
    # Uncertainty consistency
    # -------------------------------------------------------------
    uncertainty = uncertainty_consistency_loss(
        output["logits"],
        target,
    )

    # -------------------------------------------------------------
    # Composite objective
    # -------------------------------------------------------------
    total = (
        1.00 * dice
        + 0.20 * stability
        + 0.20 * morphology_loss_value
        + 0.20 * knowledge_loss
        + 0.10 * topology
        + 0.10 * connectivity
        + 0.05 * smoothness
        + 0.05 * uncertainty
    )

    return total


def main():
    parser = argparse.ArgumentParser(
        description="Train KE-NTMN."
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to configuration YAML file.",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------
    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    seed = config.get(
        "reproducibility",
        {},
    ).get(
        "random_seed",
        config.get("seed", 42),
    )

    set_seed(seed)

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # -------------------------------------------------------------
    # Data configuration
    # -------------------------------------------------------------
    data_config = config["data"]

    data_root = Path(
        data_config.get(
            "root",
            "data",
        )
    )

    image_size = tuple(
        data_config.get(
            "image_size",
            [256, 256],
        )
    )

    num_classes = data_config.get(
        "num_classes",
        4,
    )

    # -------------------------------------------------------------
    # Datasets
    # -------------------------------------------------------------
    train_dataset = BrainTumorDataset(
        root=data_root / "train",
        image_size=image_size,
        num_classes=num_classes,
        augment=data_config.get(
            "augmentation",
            {},
        ).get(
            "enabled",
            True,
        ),
    )

    val_dataset = BrainTumorDataset(
        root=data_root / "val",
        image_size=image_size,
        num_classes=num_classes,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"].get(
            "batch_size",
            8,
        ),
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print(
        f"Training cases: {len(train_dataset)}"
    )
    print(
        f"Validation cases: {len(val_dataset)}"
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------
    model_config = config["model"]

    model = KE_NTMN(
        num_classes=model_config.get(
            "num_classes",
            4,
        ),
        steps=model_config.get(
            "steps",
            model_config.get(
                "recurrent_steps",
                3,
            ),
        ),
        structural_dim=model_config.get(
            "structural_feature_dim",
            model_config.get(
                "structural_dim",
                256,
            ),
        ),
    ).to(device)

    # -------------------------------------------------------------
    # Optimizer
    # -------------------------------------------------------------
    training_config = config["training"]
    optimizer_config = training_config["optimizer"]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optimizer_config.get(
            "learning_rate",
            1e-4,
        ),
        weight_decay=optimizer_config.get(
            "weight_decay",
            1e-4,
        ),
    )

    # -------------------------------------------------------------
    # Learning-rate scheduler
    # -------------------------------------------------------------
    scheduler_config = training_config.get(
        "scheduler",
        {},
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config.get(
            "epochs",
            150,
        ),
        eta_min=scheduler_config.get(
            "min_learning_rate",
            1e-6,
        ),
    )

    # -------------------------------------------------------------
    # Checkpoint directory
    # -------------------------------------------------------------
    checkpoint_dir = Path(
        config.get(
            "logging",
            {},
        ).get(
            "checkpoint_dir",
            "checkpoints",
        )
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------
    epochs = training_config.get(
        "epochs",
        150,
    )

    best_val_dice = -float("inf")

    for epoch in range(1, epochs + 1):

        model.train()

        running_loss = 0.0

        for batch in train_loader:

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            target = batch["mask"].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            output = model(
                images,
                target,
            )

            loss = compute_loss(
                output,
                target,
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        train_loss = (
            running_loss / len(train_loader)
        )

        val_dice = evaluate(
            model,
            val_loader,
            device,
            num_classes,
        )

        current_lr = optimizer.param_groups[0][
            "lr"
        ]

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"Loss: {train_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        # ---------------------------------------------------------
        # Save best validation checkpoint
        # ---------------------------------------------------------
        if val_dice > best_val_dice:

            best_val_dice = val_dice

            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_dice": best_val_dice,
                "seed": seed,
                "config": config,
            }

            torch.save(
                checkpoint,
                checkpoint_dir / "best.pt",
            )

            print(
                f"Saved best checkpoint: "
                f"{checkpoint_dir / 'best.pt'}"
            )

    print("\nTraining complete.")
    print(
        f"Best validation Dice: "
        f"{best_val_dice:.4f}"
    )


if __name__ == "__main__":
    main()