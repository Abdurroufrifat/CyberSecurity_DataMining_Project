# CyberSecurity Data Mining Project

Machine-learning intrusion detection experiments on the **CICIDS2017** dataset, with an extended research pipeline focused on data leakage, temporal generalization, unseen attacks, reproducibility, and model explainability.

This repository contains the original cybersecurity data-mining project together with an upgraded experimental pipeline designed to test whether strong intrusion-detection performance remains reliable under more realistic train/test conditions.

## Project Overview

Machine-learning intrusion detection models can achieve very high scores on CICIDS2017 when conventional random train/test splits are used. However, random splitting can allow very similar network flows to appear in both training and testing data.

This project therefore evaluates intrusion detection under several conditions, including random, grouped, temporal, temporal-novel, and unseen-attack experiments.

The project includes:

- CICIDS2017 data loading and preprocessing
- Dataset cleaning and auditing
- Duplicate flow-fingerprint analysis
- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- Random train/test evaluation
- Leakage-resistant grouped evaluation
- Temporal generalization experiments
- Unseen attack-family experiments
- PR-AUC, macro-F1, MCC, FPR, FNR, recall, and other metrics
- SHAP-based explainability
- Publication tables and figures
- Reproducibility metadata and split manifests

## Repository Structure

```text
CyberSecurity_DataMining_Project/
│
├── main.py
├── src/
├── figures/
├── reports/
├── publication_results/
├── publication_results_fixed/
│
├── sci_ids_upgrade/
│   ├── main.py
│   ├── requirements.txt
│   ├── PROJECT_AUDIT.md
│   ├── README.md
│   ├── ids_sci/
│   ├── tests/
│   ├── outputs/
│   └── outputs_fullscale/
│
├── CICIDS2017_SCI_Upgrade_v1.1.zip
├── CICIDS2017_SCI_Upgrade_v1.2.zip
├── LICENSE
└── README.md
```

The root `main.py` belongs to the original dataset-processing workflow.

The main research-oriented pipeline is located in:

```text
sci_ids_upgrade/
```

## Dataset

The experiments use the **CICIDS2017** intrusion-detection dataset.

For the upgraded pipeline, place the eight original `MachineLearningCSV` files inside a `dataset` directory:

```text
dataset/
├── Monday-WorkingHours.pcap_ISCX.csv
├── Tuesday-WorkingHours.pcap_ISCX.csv
├── Wednesday-workingHours.pcap_ISCX.csv
├── Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
├── Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
├── Friday-WorkingHours-Morning.pcap_ISCX.csv
├── Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
└── Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

The CICIDS2017 dataset is not distributed under this repository's MIT license. Dataset use remains subject to the terms of its original publisher.

## Environment

Python **3.11 or Python 3.12** is recommended.

Clone the repository:

```powershell
git clone https://github.com/Abdurroufrifat/CyberSecurity_DataMining_Project.git
```

Open the upgraded project:

```powershell
cd CyberSecurity_DataMining_Project\sci_ids_upgrade
```

Create a virtual environment:

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents environment activation, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Main dependencies include:

```text
pandas
numpy
scikit-learn
xgboost
matplotlib
scipy
pyarrow
joblib
shap
```

## Dataset Audit

Before training the models, run the dataset audit:

```powershell
.\.venv\Scripts\python.exe main.py audit --dataset-dir dataset --output-dir outputs
```

The audit examines:

- dataset cleaning
- class distributions
- capture-day distributions
- feature inventory
- duplicate flow fingerprints
- conflicting labels
- cross-source fingerprint overlap

The project audit identified:

```text
Cleaned rows:                    2,572,435
Numeric features:               78
Official CICIDS2017 labels:     15
Cross-source fingerprints:      46,546
Conflicting fingerprints:       718
```

These checks help identify possible leakage between training and testing data.

## Run the Experiments

For a laptop-sized validation experiment:

```powershell
.\.venv\Scripts\python.exe main.py run `
  --dataset-dir dataset `
  --output-dir outputs `
  --protocols random group temporal temporal_novel unseen `
  --models logistic_regression decision_tree random_forest xgboost `
  --seeds 42 52 62 `
  --max-train-rows 600000 `
  --max-test-rows 200000
```

For a full uncapped experiment:

```powershell
.\.venv\Scripts\python.exe main.py run --dataset-dir dataset --output-dir outputs_fullscale --protocols random group temporal temporal_novel unseen --models logistic_regression decision_tree random_forest xgboost --seeds 42 52 62 72 82 --max-train-rows 0 --max-test-rows 0 --explain
```

A value of `0` means that no sampling limit is applied.

## Evaluation Protocols

### Random

Conventional stratified random train/test split.

It is useful as a baseline but can produce optimistic results when similar network flows appear in both partitions.

### Group

Feature-fingerprint groups are prevented from appearing in both training and testing data.

This helps reduce duplicate-flow leakage.

### Temporal

The model is trained on:

```text
Monday → Thursday
```

and tested on:

```text
Friday
```

This evaluates performance on later network traffic.

### Temporal Novel

This extends the temporal experiment by removing Friday flows whose exact feature fingerprints already appeared during Monday through Thursday.

### Unseen Attack

One attack family is removed from training and evaluated separately.

This experiment studies model behavior when a specific attack family was absent from the training data.

## Verified Experimental Results

Saved project results include the following sampled XGBoost experiments:

| Evaluation | Accuracy | Attack Precision | Attack Recall | Attack F1 |
|---|---:|---:|---:|---:|
| Stratified random split | 0.998783 | 0.993486 | 0.999194 | 0.996331 |
| Monday-Thursday → Friday | 0.776160 | 0.998469 | 0.368320 | 0.538131 |

The temporal experiment produced much lower attack recall than the random split.

This difference is why the upgraded project does not rely only on random-split accuracy when evaluating intrusion-detection performance.

These numbers come from saved sampled experiments in this repository and should not be treated as universal CICIDS2017 benchmark values.

## Generate Publication Results

After running the experiments:

```powershell
.\.venv\Scripts\python.exe main.py report --output-dir outputs
```

Generated results include:

```text
outputs/
│
├── audit/
│   ├── cleaning_summary.csv
│   ├── class_distribution.csv
│   ├── day_class_distribution.csv
│   ├── feature_inventory.csv
│   └── fingerprint_audit.csv
│
├── raw_metrics.csv
├── aggregate_metrics.csv
├── split_manifest.csv
├── tables/
├── figures/
├── explanations/
└── run_metadata.json
```

## Evaluation Metrics

The experiments consider multiple metrics rather than accuracy alone.

Important metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Macro-F1
- PR-AUC
- MCC
- False Positive Rate
- False Negative Rate
- Balanced Accuracy

This is particularly important for cybersecurity datasets where attack classes may be highly imbalanced.

## Model Explainability

The upgraded pipeline includes SHAP-based analysis for supported machine-learning models.

Explainability results are stored under:

```text
outputs/explanations/
```

Feature importance should be interpreted together with the evaluation protocol because a feature that performs well under random splitting may behave differently under temporal or unseen-attack conditions.

## Reproducibility

The upgraded pipeline records:

```text
run_metadata.json
split_manifest.csv
raw_metrics.csv
aggregate_metrics.csv
```

These files make it easier to inspect:

- dataset sizes
- experiment seeds
- evaluation protocols
- train/test split conditions
- sampling limits
- model configurations
- reported metrics

More implementation details are available in:

```text
sci_ids_upgrade/README.md
sci_ids_upgrade/PROJECT_AUDIT.md
```

## Research Goal

The purpose of this project is not simply to obtain the highest possible CICIDS2017 classification accuracy.

The main research question is whether intrusion-detection performance remains strong after reducing data leakage and evaluating models against later or previously unseen network traffic.

## Author

**Abdur Rouf**

GitHub:  
https://github.com/Abdurroufrifat

Repository:  
https://github.com/Abdurroufrifat/CyberSecurity_DataMining_Project

## License

This project is licensed under the **MIT License**.

See:

```text
LICENSE
```

for the full license text.

Third-party datasets and external resources remain subject to their own licenses and usage terms.
