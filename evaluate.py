import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.ketmn.data import BrainTumorDataset
from src.ketmn.model import KE_NTMN


def dice_score(pred, target, num_classes=4):
    """Compute mean foreground Dice for one patient."""

    scores = []

    for class_id in range(1, num_classes):
        pred_class = pred == class_id
        target_class = target == class_id

        denominator = pred_class.sum() + target_class.sum()

        if denominator == 0:
            scores.append(1.0)
        else:
            intersection = np.logical_and(
                pred_class,
                target_class,
            ).sum()

            scores.append(
                2.0 * intersection / denominator
            )

    return float(np.mean(scores))


def load_checkpoint(model, checkpoint_path, device):
    """Load a saved KE-NTMN checkpoint."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate KE-NTMN on a dataset split."
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to configuration YAML file.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained model checkpoint.",
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
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

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------
    data_root = config["data"].get(
        "root",
        "data",
    )

    dataset = BrainTumorDataset(
        root=Path(data_root) / args.split,
        image_size=tuple(
            config["data"].get(
                "image_size",
                [256, 256],
            )
        ),
        num_classes=config["data"].get(
            "num_classes",
            4,
        ),
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
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

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    model = load_checkpoint(
        model,
        checkpoint_path,
        device,
    )

    model.eval()

    # -------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------
    patient_scores = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            target = batch["mask"].cpu().numpy()[0]

            output = model(images)

            prediction = (
                output["prob"]
                .argmax(dim=1)
                .cpu()
                .numpy()[0]
            )

            score = dice_score(
                prediction,
                target,
                num_classes=model_config.get(
                    "num_classes",
                    4,
                ),
            )

            patient_scores.append(score)

            case_id = batch["case_id"][0]

            print(
                f"{case_id}: Dice = {score:.4f}"
            )

    # -------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------
    if not patient_scores:
        raise RuntimeError(
            "No patients were found in the selected split."
        )

    scores = np.asarray(
        patient_scores,
        dtype=np.float64,
    )

    print("\nEvaluation Summary")
    print("------------------")
    print(f"Patients: {len(scores)}")
    print(f"Mean Dice: {scores.mean():.4f}")
    print(f"Std Dice:  {scores.std():.4f}")


if __name__ == "__main__":
    main()