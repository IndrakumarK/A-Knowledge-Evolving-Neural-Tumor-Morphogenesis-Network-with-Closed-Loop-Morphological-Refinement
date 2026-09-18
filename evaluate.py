import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.ketmn.data import BrainTumorDataset
from src.ketmn.model import KE_NTMN

DATASET_VALID_CLASSES = {
    "UPenn": np.array([True, True, True, True], dtype=bool),
    "UPenn-GBM": np.array([True, True, True, True], dtype=bool),
    "TCGA": np.array([True, True, True, True], dtype=bool),
    "TCGA-GBM": np.array([True, True, True, True], dtype=bool),
    "MOTUM": np.array([True, True, False, True], dtype=bool),
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------
def get_dataset_name(batch, default_name=None):
    """Obtain dataset name from the batch."""

    if "dataset" in batch:
        dataset = batch["dataset"]

        if isinstance(dataset, (list, tuple)):
            return str(dataset[0])

        return str(dataset)

    if default_name is not None:
        return str(default_name)

    return None


def get_valid_classes(batch, dataset_name, num_classes):
   
    if "valid_classes" in batch:
        valid_classes = batch["valid_classes"]

        if torch.is_tensor(valid_classes):
            valid_classes = valid_classes.cpu().numpy()

        valid_classes = np.asarray(valid_classes)

        if valid_classes.ndim > 1:
            valid_classes = valid_classes[0]

        return valid_classes.astype(bool)

    if dataset_name in DATASET_VALID_CLASSES:
        valid_classes = DATASET_VALID_CLASSES[dataset_name]

        if len(valid_classes) != num_classes:
            raise ValueError(
                f"Valid-class definition for {dataset_name} has "
                f"{len(valid_classes)} classes, but model uses "
                f"{num_classes} classes."
            )

        return valid_classes

        return np.ones(num_classes, dtype=bool)


def _surface_distance_hd95(pred_mask, target_mask):
  
    from scipy.ndimage import binary_erosion, distance_transform_edt

    pred_mask = np.asarray(pred_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)

    pred_empty = not pred_mask.any()
    target_empty = not target_mask.any()

    if pred_empty and target_empty:
        return 0.0

    if pred_empty or target_empty:
        return float("inf")

    pred_surface = pred_mask ^ binary_erosion(
        pred_mask,
        border_value=0,
    )

    target_surface = target_mask ^ binary_erosion(
        target_mask,
        border_value=0,
    )

    if not pred_surface.any() or not target_surface.any():
        return 0.0

    target_distance = distance_transform_edt(
        ~target_surface
    )

    pred_distance = distance_transform_edt(
        ~pred_surface
    )

    pred_to_target = target_distance[pred_surface]
    target_to_pred = pred_distance[target_surface]

    distances = np.concatenate(
        [
            pred_to_target,
            target_to_pred,
        ]
    )

    return float(np.percentile(distances, 95))


def compute_patient_metrics(
    prediction,
    target,
    valid_classes,
    num_classes=4,
):
    dice_scores = []
    iou_scores = []
    precision_scores = []
    recall_scores = []
    specificity_scores = []
    hd95_scores = []

    for class_id in range(1, num_classes):

            if not valid_classes[class_id]:
            continue

        pred_class = prediction == class_id
        target_class = target == class_id

        tp = np.logical_and(
            pred_class,
            target_class,
        ).sum()

        fp = np.logical_and(
            pred_class,
            ~target_class,
        ).sum()

        fn = np.logical_and(
            ~pred_class,
            target_class,
        ).sum()

        tn = np.logical_and(
            ~pred_class,
            ~target_class,
        ).sum()

        # -------------------------------------------------------------
        # Dice
        # -------------------------------------------------------------
        denominator = (
            pred_class.sum()
            + target_class.sum()
        )

        if denominator == 0:
            dice = 1.0
        else:
            dice = (
                2.0 * tp / denominator
            )

        # -------------------------------------------------------------
        # IoU
        # -------------------------------------------------------------
        union = (
            pred_class
            | target_class
        ).sum()

        if union == 0:
            iou = 1.0
        else:
            iou = tp / union

        # -------------------------------------------------------------
        # Precision
        # -------------------------------------------------------------
        precision_denominator = tp + fp

        if precision_denominator == 0:
            precision = 1.0
        else:
            precision = (
                tp / precision_denominator
            )

        # -------------------------------------------------------------
        # Recall / Sensitivity
        # -------------------------------------------------------------
        recall_denominator = tp + fn

        if recall_denominator == 0:
            recall = 1.0
        else:
            recall = (
                tp / recall_denominator
            )

        # -------------------------------------------------------------
        # Specificity
        # -------------------------------------------------------------
        specificity_denominator = tn + fp

        if specificity_denominator == 0:
            specificity = 1.0
        else:
            specificity = (
                tn / specificity_denominator
            )

        # -------------------------------------------------------------
        # HD95
        # -------------------------------------------------------------
        hd95 = _surface_distance_hd95(
            pred_class,
            target_class,
        )

        dice_scores.append(dice)
        iou_scores.append(iou)
        precision_scores.append(precision)
        recall_scores.append(recall)
        specificity_scores.append(specificity)
        hd95_scores.append(hd95)

    if not dice_scores:
        raise RuntimeError(
            "No valid foreground classes were available "
            "for quantitative evaluation."
        )

    return {
        "dice": float(np.mean(dice_scores)),
        "iou": float(np.mean(iou_scores)),
        "hd95": float(np.mean(hd95_scores)),
        "precision": float(np.mean(precision_scores)),
        "recall": float(np.mean(recall_scores)),
        "specificity": float(np.mean(specificity_scores)),
    }


# ---------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------
def load_checkpoint(
    model,
    checkpoint_path,
    device,
):
    """Load a saved KE-NTMN checkpoint."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict):

        if "model" in checkpoint:
            state_dict = checkpoint["model"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Handle checkpoints saved using DataParallel.
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    return model


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate KE-NTMN on a patient-level dataset split."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to configuration YAML file.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained KE-NTMN checkpoint.",
    )

    parser.add_argument(
        "--split",
        default="test",
        choices=[
            "train",
            "val",
            "test",
        ],
        help="Dataset split to evaluate.",
    )

    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Dataset name, e.g. UPenn-GBM, TCGA-GBM, "
            "or MOTUM. If omitted, dataset metadata "
            "from the dataset is used."
        ),
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------
    config_path = Path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    # -----------------------------------------------------------------
    # Device
    # -----------------------------------------------------------------
    configured_device = config.get(
        "training",
        {},
    ).get(
        "device",
        "cuda",
    )

    if (
        configured_device == "cuda"
        and torch.cuda.is_available()
    ):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(
        f"Using device: {device}"
    )

    # -----------------------------------------------------------------
    # Dataset configuration
    # -----------------------------------------------------------------
    data_config = config.get(
        "data",
        {},
    )

    data_root = data_config.get(
        "root",
        "data",
    )

    image_size = tuple(
        data_config.get(
            "image_size",
            [256, 256],
        )
    )

    num_classes = int(
        data_config.get(
            "num_classes",
            4,
        )
    )

    # -----------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------
    dataset = BrainTumorDataset(
        root=Path(data_root) / args.split,
        image_size=image_size,
        num_classes=num_classes,
        augment=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------
    model_config = config.get(
        "model",
        {},
    )

    recurrent_steps = model_config.get(
        "steps",
        model_config.get(
            "recurrent_steps",
            3,
        ),
    )

    structural_dim = model_config.get(
        "structural_feature_dim",
        model_config.get(
            "structural_dim",
            256,
        ),
    )

    model = KE_NTMN(
        num_classes=num_classes,
        steps=recurrent_steps,
        structural_dim=structural_dim,
    ).to(device)

    checkpoint_path = Path(
        args.checkpoint
    )

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

    # -----------------------------------------------------------------
    # Evaluation
    # -----------------------------------------------------------------
    patient_results = []

    with torch.no_grad():

        for batch in loader:

            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            target = (
                batch["mask"]
                .cpu()
                .numpy()[0]
            )

            dataset_name = get_dataset_name(
                batch,
                default_name=args.dataset,
            )

            if dataset_name is None:
                raise ValueError(
                    "Dataset name is unavailable. "
                    "Provide --dataset or return "
                    "'dataset' from BrainTumorDataset."
                )

            valid_classes = get_valid_classes(
                batch,
                dataset_name,
                num_classes,
            )

            # ---------------------------------------------------------
            # Final KE-NTMN prediction
            # ---------------------------------------------------------
            output = model(images)

            if "prob" not in output:
                raise KeyError(
                    "KE_NTMN output must contain "
                    "the final probability map under "
                    "output['prob']."
                )

            probability = output["prob"]

            prediction = (
                probability
                .argmax(dim=1)
                .cpu()
                .numpy()[0]
            )

            # ---------------------------------------------------------
            # Patient-level metrics
            # ---------------------------------------------------------
            metrics = compute_patient_metrics(
                prediction=prediction,
                target=target,
                valid_classes=valid_classes,
                num_classes=num_classes,
            )

            case_id = batch["case_id"][0]

            result = {
                "case_id": case_id,
                "dataset": dataset_name,
                **metrics,
            }

            patient_results.append(result)

            print(
                f"{case_id} | "
                f"{dataset_name} | "
                f"Dice={metrics['dice']:.4f} | "
                f"IoU={metrics['iou']:.4f} | "
                f"HD95={metrics['hd95']:.4f} | "
                f"Precision={metrics['precision']:.4f} | "
                f"Recall={metrics['recall']:.4f} | "
                f"Specificity={metrics['specificity']:.4f}"
            )

    # -----------------------------------------------------------------
    # Check results
    # -----------------------------------------------------------------
    if not patient_results:
        raise RuntimeError(
            "No patients were found in the selected split."
        )

    # -----------------------------------------------------------------
    # Overall patient-level statistics
    # -----------------------------------------------------------------
    metric_names = [
        "dice",
        "iou",
        "hd95",
        "precision",
        "recall",
        "specificity",
    ]

    print("\n")
    print("=" * 72)
    print("KE-NTMN PATIENT-LEVEL EVALUATION")
    print("=" * 72)

    print(
        f"Patients evaluated: "
        f"{len(patient_results)}"
    )

    print(
        f"Split: {args.split}"
    )

    if args.dataset:
        print(
            f"Dataset: {args.dataset}"
        )

    print("-" * 72)

    for metric in metric_names:

        values = np.asarray(
            [
                result[metric]
                for result in patient_results
            ],
            dtype=np.float64,
        )

         finite_values = values[
            np.isfinite(values)
        ]

        if len(finite_values) == 0:
            mean_value = float("inf")
            std_value = float("inf")
        else:
            mean_value = finite_values.mean()
            std_value = finite_values.std(
                ddof=0
            )

        print(
            f"{metric.capitalize():<15}: "
            f"{mean_value:.4f} ± "
            f"{std_value:.4f}"
        )

    # -----------------------------------------------------------------
    # Dataset-wise summaries
    # -----------------------------------------------------------------
    datasets = sorted(
        set(
            result["dataset"]
            for result in patient_results
        )
    )

    if len(datasets) > 1:

        print("\n")
        print("=" * 72)
        print("DATASET-WISE SUMMARY")
        print("=" * 72)

        for dataset_name in datasets:

            subset = [
                result
                for result in patient_results
                if result["dataset"] == dataset_name
            ]

            print(
                f"\n{dataset_name}"
            )
            print(
                f"Patients: {len(subset)}"
            )

            for metric in metric_names:

                values = np.asarray(
                    [
                        result[metric]
                        for result in subset
                    ],
                    dtype=np.float64,
                )

                finite_values = values[
                    np.isfinite(values)
                ]

                if len(finite_values) == 0:
                    mean_value = float("inf")
                    std_value = float("inf")
                else:
                    mean_value = finite_values.mean()
                    std_value = finite_values.std(
                        ddof=0
                    )

                print(
                    f"  {metric.capitalize():<13}: "
                    f"{mean_value:.4f} ± "
                    f"{std_value:.4f}"
                )


if __name__ == "__main__":
    main()
