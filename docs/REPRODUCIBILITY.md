# Reproducibility Checklist

The following checklist should be completed before reproducing or extending the KE-NTMN experiments.

## Dataset and Splits

- [ ] Record the exact dataset name and version used for each experiment.
- [ ] Record the exact patient-level training, validation, and test partitions.
- [ ] Ensure that no patient occurs in more than one partition.
- [ ] Preserve the same patient-level partitions across comparative experiments.
- [ ] Do not redistribute UPenn-GBM, TCGA-GBM, MOTUM, or private clinical MRI data through this repository.

## Preprocessing and Training

- [ ] Record all preprocessing and data-conversion steps.
- [ ] Record modality-specific normalization procedures.
- [ ] Record all data-augmentation operations and their parameters.
- [ ] Record the random seed used for training and evaluation.
- [ ] Record the exact PyTorch version and other relevant software dependencies.
- [ ] Record the actual GPU model and available memory used for experiments.
- [ ] Keep the input resolution and model configuration consistent with the reported experiments.
- [ ] Select model checkpoints using validation data only.
- [ ] Do not use test-set performance for model selection or hyperparameter tuning.

## Evaluation and Results

- [ ] Save patient-level predictions for each evaluated method.
- [ ] Save patient-level metric values used to calculate reported summary statistics.
- [ ] Compute pooled statistics directly from patient-level observations rather than averaging dataset-level means.
- [ ] Preserve patient-level pairing when comparing two methods on the same test cases.
- [ ] Use paired statistical tests only when the observations are appropriately paired.
- [ ] Generate bootstrap confidence intervals from the paired patient-level observations.
- [ ] Record the number of patients contributing to each statistical comparison.

## Comparator Provenance

- [ ] Label a comparator as **Reproduced** only when its implementation was actually executed under the stated experimental protocol.
- [ ] Clearly identify results obtained from published or externally reported sources.
- [ ] Do not present published results as independently reproduced results.
- [ ] Record the source and provenance of each comparative result reported in tables.

## External Clinical Cohort

- [ ] Keep the independently collected clinical MRI cohort separate from the benchmark datasets.
- [ ] Do not use the external clinical cohort for training, validation, parameter selection, or quantitative benchmark evaluation.
- [ ] Do not quantitatively evaluate cases for which voxel-level tumor annotations are unavailable.
- [ ] Use the external clinical cohort only for the qualitative evaluation described in the manuscript.
- [ ] Keep private clinical MRI data confidential and do not include them in the public repository.

## Recommended Reproducibility Records

For each experiment, retain:

```text
experiment/
├── config.yaml
├── dataset_version.txt
├── patient_splits.json
├── environment.txt
├── training_log.csv
├── checkpoint/
├── predictions/
├── patient_metrics.csv
└── statistics/
    ├── confidence_intervals.csv
    └── significance_tests.csv