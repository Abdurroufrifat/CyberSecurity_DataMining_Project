# CICIDS2017 SCI-Grade IDS Experiment Suite

This project upgrades the original class experiment into a reproducible study of
data leakage, temporal generalization, unseen-attack detection, and model
explainability on CICIDS2017.

The central comparison is not simply which classifier obtains the highest score
under a random split. It asks how much performance survives when identical flow
fingerprints are kept in one partition, Friday is held out as future traffic,
and an attack family is absent from training.

## 1. Environment

Python 3.11 or 3.12 is recommended. From this folder, create an environment and
install the dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Dataset layout

Put the eight original CICIDS2017 MachineLearningCSV files inside `dataset/`.
Derived files such as `combined_dataset.csv` and `preprocessed_dataset.csv` are
ignored automatically.

```text
dataset/
  Monday-WorkingHours.pcap_ISCX.csv
  Tuesday-WorkingHours.pcap_ISCX.csv
  Wednesday-workingHours.pcap_ISCX.csv
  Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv
  Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv
  Friday-WorkingHours-Morning.pcap_ISCX.csv
  Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
  Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
```

If `sci_ids_upgrade` is inside your existing project and the original dataset
is its sibling, keep the data where it is and use `--dataset-dir ..\dataset` in
every command. This avoids copying the large CSV files.

## 3. Audit and cache the dataset

```powershell
.\.venv\Scripts\python.exe main.py audit --dataset-dir dataset --output-dir outputs
```

This writes cleaning counts, class/day distributions, the real feature
inventory, duplicate-fingerprint diagnostics, and a compact Parquet cache. The
cache retains `Source_File` and `Capture_Day`; these fields are never supplied
to a model.

## 4. Run a laptop-sized validation experiment

Start with this command on a 16 GB RAM laptop:

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

On Command Prompt, place the command on one line and remove the PowerShell
backticks.

After the validation run succeeds, the uncapped experiment is:

```powershell
.\.venv\Scripts\python.exe main.py run --dataset-dir dataset --output-dir outputs_full --protocols random group temporal temporal_novel unseen --models logistic_regression decision_tree random_forest xgboost --seeds 42 52 62 72 82 --max-train-rows 0 --max-test-rows 0 --explain
```

`0` means no sampling cap. If Random Forest exceeds available memory, rerun the
full experiment without it and retain its capped result as a secondary
baseline. Do not silently mix capped and uncapped results in one comparison.

## 5. Build publication tables and figures

```powershell
.\.venv\Scripts\python.exe main.py report --output-dir outputs
```

Important outputs include:

```text
outputs/
  audit/
    cleaning_summary.csv
    class_distribution.csv
    day_class_distribution.csv
    feature_inventory.csv
    fingerprint_audit.csv
  raw_metrics.csv
  aggregate_metrics.csv
  split_manifest.csv
  tables/
  figures/
  explanations/
  run_metadata.json
```

## Experiment protocols

- `random`: conventional stratified row split. This is the optimistic baseline.
- `group`: feature-fingerprint groups cannot cross the train/test boundary.
- `temporal`: Monday-Thursday training and Friday testing.
- `temporal_novel`: temporal split after removing Friday test flows whose exact
  feature fingerprints appeared during Monday-Thursday.
- `unseen`: one attack family is removed from training and evaluated with a
  disjoint benign test partition. Every held-out record is retained for rare
  families; only benign rows are reduced to meet the test-size cap.

## Reporting rules

- Report attack recall, macro-F1, PR-AUC, MCC, FPR, and FNR in addition to
  accuracy.
- Treat the random split as an upper-bound baseline, not deployment evidence.
- State all sampling caps and the number of rows in every split.
- Do not claim zero-day detection from a normal random split.
- Keep the original attack labels for the audit. Normalization only repairs
  punctuation/encoding variants in the Web Attack labels.
- Report rare-class results even when they are poor. The pipeline never silently
  deletes an attack class because it has fewer than five sampled records.
