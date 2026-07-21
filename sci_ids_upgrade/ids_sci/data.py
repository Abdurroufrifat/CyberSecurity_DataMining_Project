from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROVENANCE_COLUMNS = ["Label", "Binary_Label", "Source_File", "Capture_Day", "Feature_Fingerprint"]
DERIVED_MARKERS = (
    "combined", "preprocessed", "result", "summary", "classification_report",
    "feature_importance", "prediction", "metric",
)


@dataclass
class LoadedDataset:
    frame: pd.DataFrame
    feature_columns: list[str]
    cleaning_summary: pd.DataFrame
    source_files: list[str]


def discover_raw_csvs(dataset_dir: Path) -> list[Path]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    files = []
    for path in sorted(dataset_dir.glob("*.csv")):
        lower = path.name.lower()
        if any(marker in lower for marker in DERIVED_MARKERS):
            continue
        files.append(path)
    if not files:
        raise FileNotFoundError(f"No original CICIDS2017 CSV files found in {dataset_dir}")
    return files


def capture_day(filename: str) -> str:
    lower = filename.lower()
    for day in ["monday", "tuesday", "wednesday", "thursday", "friday"]:
        if day in lower:
            return day.capitalize()
    return "Unknown"


def normalize_label(value: object) -> str:
    text = str(value).strip()
    lowered = text.lower()
    if "web attack" in lowered:
        if "sql" in lowered:
            return "Web Attack - SQL Injection"
        if "xss" in lowered:
            return "Web Attack - XSS"
        if "brute" in lowered:
            return "Web Attack - Brute Force"
    replacements = {
        "benign": "BENIGN",
        "bot": "Bot",
        "ddos": "DDoS",
        "dos goldeneye": "DoS GoldenEye",
        "dos hulk": "DoS Hulk",
        "dos slowhttptest": "DoS Slowhttptest",
        "dos slowloris": "DoS slowloris",
        "ftp-patator": "FTP-Patator",
        "heartbleed": "Heartbleed",
        "infiltration": "Infiltration",
        "portscan": "PortScan",
        "ssh-patator": "SSH-Patator",
    }
    compact = re.sub(r"\s+", " ", lowered)
    return replacements.get(compact, text)


def _clean_one_file(path: Path, reference_features: list[str] | None) -> tuple[pd.DataFrame, dict, list[str]]:
    raw = pd.read_csv(path, low_memory=False, encoding="latin1")
    raw.columns = raw.columns.str.strip()
    raw = raw.loc[:, ~raw.columns.duplicated()].copy()
    if "Label" not in raw.columns:
        raise ValueError(f"Label column missing from {path.name}")

    candidate_columns = [
        c for c in raw.columns
        if c not in {"Label", "Flow ID", "Source IP", "Destination IP", "Timestamp"}
    ]
    numeric = pd.DataFrame(index=raw.index)
    for column in candidate_columns:
        numeric[column] = pd.to_numeric(raw[column], errors="coerce", downcast="float")

    if reference_features is None:
        feature_columns = list(numeric.columns)
    else:
        missing = sorted(set(reference_features) - set(numeric.columns))
        extra = sorted(set(numeric.columns) - set(reference_features))
        if missing or extra:
            raise ValueError(
                f"Feature schema mismatch in {path.name}; missing={missing}, extra={extra}"
            )
        feature_columns = reference_features
        numeric = numeric[feature_columns]

    numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
    labels = raw["Label"].map(normalize_label)
    valid = numeric.notna().all(axis=1) & labels.notna() & labels.ne("")
    clean = numeric.loc[valid].astype(np.float32, copy=False)
    clean["Label"] = labels.loc[valid].to_numpy()

    before_dedup = len(clean)
    clean.drop_duplicates(subset=feature_columns + ["Label"], inplace=True)
    after_dedup = len(clean)
    clean["Binary_Label"] = (clean["Label"] != "BENIGN").astype(np.int8)
    clean["Source_File"] = path.name
    clean["Capture_Day"] = capture_day(path.name)

    summary = {
        "Source_File": path.name,
        "Capture_Day": capture_day(path.name),
        "Raw_Rows": len(raw),
        "Rows_After_Nonfinite_Removal": before_dedup,
        "Rows_After_Within_Source_Deduplication": after_dedup,
        "Removed_Nonfinite_Or_Missing": len(raw) - before_dedup,
        "Removed_Within_Source_Duplicates": before_dedup - after_dedup,
        "Feature_Count": len(feature_columns),
    }
    return clean, summary, feature_columns


def add_fingerprints(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    frame["Feature_Fingerprint"] = pd.util.hash_pandas_object(
        frame[feature_columns], index=False
    ).astype("uint64")
    return frame


def load_or_build_dataset(
    dataset_dir: Path,
    cache_path: Path,
    force_rebuild: bool = False,
) -> LoadedDataset:
    pickle_cache = cache_path.with_suffix(".pkl")
    summary_path = cache_path.with_suffix(".cleaning.csv")
    metadata_path = cache_path.with_suffix(".metadata.json")
    available_cache = cache_path if cache_path.exists() else pickle_cache
    if available_cache.exists() and summary_path.exists() and metadata_path.exists() and not force_rebuild:
        frame = (
            pd.read_parquet(available_cache)
            if available_cache.suffix == ".parquet"
            else pd.read_pickle(available_cache)
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return LoadedDataset(
            frame=frame,
            feature_columns=metadata["feature_columns"],
            cleaning_summary=pd.read_csv(summary_path),
            source_files=metadata["source_files"],
        )

    files = discover_raw_csvs(dataset_dir)
    frames: list[pd.DataFrame] = []
    summaries: list[dict] = []
    features: list[str] | None = None
    for path in files:
        print(f"Loading and cleaning {path.name} ...")
        clean, summary, features = _clean_one_file(path, features)
        frames.append(clean)
        summaries.append(summary)

    assert features is not None
    frame = pd.concat(frames, ignore_index=True)
    frame = add_fingerprints(frame, features)
    frame["Label"] = frame["Label"].astype("category")
    frame["Source_File"] = frame["Source_File"].astype("category")
    frame["Capture_Day"] = frame["Capture_Day"].astype("category")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(cache_path, index=False, compression="zstd")
    except ImportError:
        print("Parquet engine unavailable; using a pickle cache instead.")
        frame.to_pickle(pickle_cache)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(summary_path, index=False)
    metadata = {"feature_columns": features, "source_files": [p.name for p in files]}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return LoadedDataset(frame, features, summary_df, metadata["source_files"])


def write_audit(loaded: LoadedDataset, audit_dir: Path) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    frame = loaded.frame
    loaded.cleaning_summary.to_csv(audit_dir / "cleaning_summary.csv", index=False)
    frame["Label"].value_counts(dropna=False).rename_axis("Label").reset_index(
        name="Count"
    ).to_csv(audit_dir / "class_distribution.csv", index=False)
    pd.crosstab(frame["Capture_Day"], frame["Label"]).to_csv(
        audit_dir / "day_class_distribution.csv"
    )
    pd.DataFrame({
        "Feature_Order": range(1, len(loaded.feature_columns) + 1),
        "Feature_Name": loaded.feature_columns,
        "Dtype": [str(frame[c].dtype) for c in loaded.feature_columns],
    }).to_csv(audit_dir / "feature_inventory.csv", index=False)

    unique_pairs = frame[["Feature_Fingerprint", "Label"]].drop_duplicates()
    label_counts = unique_pairs.groupby("Feature_Fingerprint", observed=True).size()
    source_counts = frame[["Feature_Fingerprint", "Source_File"]].drop_duplicates().groupby(
        "Feature_Fingerprint", observed=True
    ).size()
    fingerprint_audit = pd.DataFrame([{
        "Rows": len(frame),
        "Unique_Feature_Fingerprints": int(frame["Feature_Fingerprint"].nunique()),
        "Repeated_Fingerprint_Rows": int(frame.duplicated("Feature_Fingerprint", keep=False).sum()),
        "Fingerprints_In_Multiple_Source_Files": int((source_counts > 1).sum()),
        "Conflicting_Label_Fingerprints": int((label_counts > 1).sum()),
    }])
    fingerprint_audit.to_csv(audit_dir / "fingerprint_audit.csv", index=False)


def feature_frame(frame: pd.DataFrame, feature_columns: Iterable[str]) -> pd.DataFrame:
    return frame[list(feature_columns)]
