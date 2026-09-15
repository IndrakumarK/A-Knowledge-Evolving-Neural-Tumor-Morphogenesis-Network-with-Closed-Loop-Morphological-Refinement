# Data Preparation

The datasets used in the KE-NTMN study are not redistributed with this repository. Users must obtain the datasets from their respective official sources and comply with their individual access, licensing, and usage requirements.

The following datasets are used in the manuscript:

- UPenn-GBM
- TCGA-GBM
- MOTUM

Private clinical MRI data used for external qualitative evaluation are **not included** in this repository and must not be redistributed.

## Dataset Organization

The generic data loader expects the following directory structure:

```text
data/
├── train/
│   └── case_id/
│       ├── T1.npy
│       ├── T1ce.npy
│       ├── T2.npy
│       ├── FLAIR.npy
│       └── mask.npy
├── val/
│   └── case_id/
│       ├── T1.npy
│       ├── T1ce.npy
│       ├── T2.npy
│       ├── FLAIR.npy
│       └── mask.npy
└── test/
    └── case_id/
        ├── T1.npy
        ├── T1ce.npy
        ├── T2.npy
        ├── FLAIR.npy
        └── mask.npy