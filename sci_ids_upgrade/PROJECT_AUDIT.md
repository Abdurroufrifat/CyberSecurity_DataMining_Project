# Audit of the Original CICIDS2017 Project

## Verified existing results

The supplied result files show the following XGBoost outcomes on sampled data:

| Evaluation | Accuracy | Attack precision | Attack recall | Attack F1 |
| --- | ---: | ---: | ---: | ---: |
| Stratified random split | 0.998783 | 0.993486 | 0.999194 | 0.996331 |
| Monday-Thursday to Friday | 0.776160 | 0.998469 | 0.368320 | 0.538131 |

The Friday confusion matrix contains 22,364 false negatives and only 20 false
positives. The key problem is therefore failure to recognize later/unseen attack
traffic, not excessive false alarms.

The sampled multiclass experiment reports 0.998800 accuracy but only 0.916141
macro-F1. Web Attack - XSS recall is 0.285714 and Bot recall is 0.744681. Three
official classes are absent from the saved 12-class result because the earlier
script discarded classes with fewer than five rows after random sampling.

## Code defects affecting reproducibility

1. The original `main.py` and preprocessing script disagree about whether
   `combined_dataset.csv` is saved in the root or the dataset directory.
2. An earlier run loaded a derived combined CSV together with the eight raw
   files, producing 5,661,486 rows and 144 columns with duplicate `Label`
   columns. The current filtering logic was added later, but this failure should
   be documented and the derived file must never be included again.
3. The original feature-engineering script creates a 1.65 GB
   `processed_data.pkl`, while the training script ignores it and independently
   recreates another split.
4. Every model receives standardized features, although trees and XGBoost do
   not require scaling.
5. The publication script calls `drop_duplicates()` after adding
   `Source_File`. Consequently, identical feature rows in different capture
   files are not treated as duplicates and can remain across evaluation
   partitions.
6. The earlier multiclass script silently removes rare classes after sampling.
7. Only one seed and no validation/tuning protocol are reported.
8. ROC-AUC plots exist for the class experiment, but ROC-AUC, PR-AUC, MCC,
   balanced accuracy, FPR, FNR, calibration, and uncertainty are missing from
   the primary result tables.
9. Built-in XGBoost importance and SHAP disagree on the dominant features. This
   is not automatically an error, but the paper must identify the importance
   definition and analyze stability across splits.

## Publication narrative

The defensible research contribution is a leakage-resistant evaluation of
generalization, not another claim of near-perfect random-split accuracy. The
replacement pipeline therefore preserves source/day provenance, constructs
feature fingerprints, reports cross-partition overlap, compares conventional
and grouped splits, performs temporal and temporal-novel tests, and evaluates
attack families completely absent from training.

The real-data audit confirmed 2,572,435 cleaned within-source rows, 78 numeric
features, all 15 official labels, 46,546 fingerprints appearing in multiple
source files, and 718 fingerprints associated with conflicting labels. Grouped
and temporal protocols remove conflicting fingerprints before training. The
unseen protocol retains every rare held-out attack record when applying a test
size cap.

Protocol-specific split diagnostics are written through a schema-safe logger so
random-overlap, conflict-removal, temporal-overlap, and held-out-family fields
remain readable in one split manifest.
