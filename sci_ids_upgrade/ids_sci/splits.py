from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    metadata: dict = field(default_factory=dict)


def remove_conflicting_fingerprints(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    unique_pairs = frame[["Feature_Fingerprint", "Label"]].drop_duplicates()
    counts = unique_pairs.groupby("Feature_Fingerprint", observed=True).size()
    bad = counts[counts > 1].index
    if len(bad) == 0:
        return frame, 0
    mask = frame["Feature_Fingerprint"].isin(bad)
    removed = int(mask.sum())
    return frame.loc[~mask].copy(), removed


def cap_stratified(frame: pd.DataFrame, max_rows: int, label_col: str, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    selected, _ = train_test_split(
        frame,
        train_size=max_rows,
        random_state=seed,
        stratify=frame[label_col],
    )
    return selected.copy()


def random_split(frame: pd.DataFrame, label_col: str, seed: int, test_size: float = 0.2) -> SplitResult:
    train, test = train_test_split(
        frame,
        test_size=test_size,
        random_state=seed,
        stratify=frame[label_col],
    )
    overlap = len(set(train["Feature_Fingerprint"]).intersection(set(test["Feature_Fingerprint"])))
    return SplitResult(train.copy(), test.copy(), {"Cross_Partition_Fingerprint_Overlap": overlap})


def _mixed_hash(values: pd.Series, seed: int) -> np.ndarray:
    x = values.to_numpy(dtype=np.uint64, copy=False)
    salt = np.uint64((seed + 1) * 0x9E3779B1)
    return x ^ salt


def group_split(frame: pd.DataFrame, seed: int, test_size: float = 0.2) -> SplitResult:
    clean, removed = remove_conflicting_fingerprints(frame)
    mixed = _mixed_hash(clean["Feature_Fingerprint"], seed)
    threshold = int(round(test_size * 10_000))
    is_test = (mixed % np.uint64(10_000)) < threshold
    train = clean.loc[~is_test].copy()
    test = clean.loc[is_test].copy()
    return SplitResult(train, test, {
        "Removed_Conflicting_Label_Rows": removed,
        "Cross_Partition_Fingerprint_Overlap": 0,
    })


def temporal_split(frame: pd.DataFrame, novel_only: bool = False) -> SplitResult:
    clean, removed_conflicts = remove_conflicting_fingerprints(frame)
    train = clean.loc[clean["Capture_Day"] != "Friday"].copy()
    test = clean.loc[clean["Capture_Day"] == "Friday"].copy()
    train_hashes = set(train["Feature_Fingerprint"].unique())
    overlap_mask = test["Feature_Fingerprint"].isin(train_hashes)
    overlapping_test_rows = int(overlap_mask.sum())
    if novel_only:
        test = test.loc[~overlap_mask].copy()
    return SplitResult(train, test, {
        "Removed_Conflicting_Label_Rows": removed_conflicts,
        "Temporal_Test_Overlap_Rows": overlapping_test_rows,
        "Temporal_Novel_Only": bool(novel_only),
    })


def unseen_attack_split(frame: pd.DataFrame, held_out_label: str, seed: int) -> SplitResult:
    clean, removed = remove_conflicting_fingerprints(frame)
    benign = clean.loc[clean["Label"] == "BENIGN"].copy()
    benign_split = group_split(benign, seed=seed, test_size=0.2)
    held_out = clean.loc[clean["Label"] == held_out_label].copy()
    other_attacks = clean.loc[
        (clean["Label"] != "BENIGN") & (clean["Label"] != held_out_label)
    ].copy()
    train = pd.concat([benign_split.train, other_attacks], ignore_index=True)
    test = pd.concat([benign_split.test, held_out], ignore_index=True)

    train_hashes = set(train["Feature_Fingerprint"].unique())
    overlap = test["Feature_Fingerprint"].isin(train_hashes)
    removed_overlap = int(overlap.sum())
    test = test.loc[~overlap].copy()
    return SplitResult(train, test, {
        "Held_Out_Attack": held_out_label,
        "Removed_Conflicting_Label_Rows": removed,
        "Removed_Test_Overlap_Rows": removed_overlap,
        "Cross_Partition_Fingerprint_Overlap": 0,
    })

