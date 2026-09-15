from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class BrainTumorDataset(Dataset):
    
    MODALITIES = ("T1", "T1ce", "T2", "FLAIR")

    def __init__(
        self,
        root,
        image_size=(256, 256),
        num_classes=4,
        augment=False,
    ):
        self.root = Path(root)
        self.image_size = tuple(image_size)
        self.num_classes = num_classes
        self.augment = augment

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.root}"
            )

        self.cases = sorted(
            case_dir
            for case_dir in self.root.iterdir()
            if case_dir.is_dir()
        )

        if not self.cases:
            raise RuntimeError(
                f"No case directories found in: {self.root}"
            )

    def __len__(self):
        return len(self.cases)

    def _load_case(self, case_dir):
        images = []

        for modality in self.MODALITIES:
            path = case_dir / f"{modality}.npy"

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing modality file: {path}"
                )

            image = np.load(path).astype(np.float32)

            if image.ndim != 2:
                raise ValueError(
                    f"{path} must contain a 2-D array, "
                    f"got shape {image.shape}"
                )

            images.append(image)

        mask_path = case_dir / "mask.npy"

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Missing segmentation mask: {mask_path}"
            )

        mask = np.load(mask_path)

        if mask.ndim != 2:
            raise ValueError(
                f"{mask_path} must contain a 2-D array, "
                f"got shape {mask.shape}"
            )

        return np.stack(images, axis=0), mask

    @staticmethod
    def _zscore(image):
        mean = image.mean()
        std = image.std()

        if std < 1e-8:
            return image - mean

        return (image - mean) / std

    def _normalize(self, images):
        normalized = np.empty_like(images, dtype=np.float32)

        for channel in range(images.shape[0]):
            normalized[channel] = self._zscore(images[channel])

        return normalized

    def _augment(self, images, mask):
        if np.random.rand() < 0.5:
            images = np.flip(images, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()

        if np.random.rand() < 0.5:
            images = np.flip(images, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()

        if np.random.rand() < 0.5:
            k = np.random.randint(1, 4)
            images = np.rot90(images, k=k, axes=(1, 2)).copy()
            mask = np.rot90(mask, k=k, axes=(0, 1)).copy()

        return images, mask

    def __getitem__(self, index):
        case_dir = self.cases[index]

        images, mask = self._load_case(case_dir)

        images = self._normalize(images)

        if self.augment:
            images, mask = self._augment(images, mask)

        if images.shape[1:] != self.image_size:
            raise ValueError(
                f"Expected image size {self.image_size}, "
                f"but case '{case_dir.name}' has "
                f"image size {images.shape[1:]}. "
                "Resize/resample the data during preprocessing."
            )

        mask = mask.astype(np.int64)

        unique_labels = np.unique(mask)

        if np.any(unique_labels < 0) or np.any(
            unique_labels >= self.num_classes
        ):
            raise ValueError(
                f"Invalid labels in {case_dir / 'mask.npy'}: "
                f"{unique_labels.tolist()}. "
                f"Expected labels in [0, {self.num_classes - 1}]."
            )

        images = torch.from_numpy(images)
        mask = torch.from_numpy(mask)

        return {
            "image": images,
            "mask": mask,
            "case_id": case_dir.name,
        }