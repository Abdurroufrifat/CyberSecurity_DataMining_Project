from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


IDENTIFIERS = ["Task", "Protocol", "Model", "Held_Out_Attack"]


def aggregate_results(raw: pd.DataFrame) -> pd.DataFrame:
    identifiers = [c for c in IDENTIFIERS if c in raw.columns]
    excluded = set(identifiers + ["Seed", "Run_ID"])
    numeric = [c for c in raw.select_dtypes(include=[np.number]).columns if c not in excluded]
    grouped = raw.groupby(identifiers, dropna=False, observed=True)
    rows = []
    for keys, block in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(identifiers, keys))
        row["Runs"] = len(block)
        for column in numeric:
            values = block[column].dropna().astype(float)
            if values.empty:
                continue
            mean = values.mean()
            std = values.std(ddof=1) if len(values) > 1 else 0.0
            half = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
            row[f"{column}_Mean"] = mean
            row[f"{column}_SD"] = std
            row[f"{column}_CI95_Low"] = mean - half
            row[f"{column}_CI95_High"] = mean + half
        rows.append(row)
    return pd.DataFrame(rows)


def build_reports(output_dir: Path) -> None:
    raw_path = output_dir / "raw_metrics.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Run experiments first; missing {raw_path}")
    raw = pd.read_csv(raw_path)
    aggregate = aggregate_results(raw)
    aggregate.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    binary = aggregate.loc[aggregate["Task"] == "binary"].copy()
    if not binary.empty:
        columns = [
            "Protocol", "Model", "Held_Out_Attack", "Runs",
            "Accuracy_Mean", "Balanced_Accuracy_Mean", "Attack_Recall_Mean",
            "F1_Mean", "MCC_Mean", "PR_AUC_Mean", "FPR_Mean", "FNR_Mean",
        ]
        columns = [c for c in columns if c in binary.columns]
        binary[columns].to_csv(tables / "binary_model_comparison.csv", index=False)

        ordinary = binary.loc[binary["Held_Out_Attack"].isna()].copy()
        if not ordinary.empty and "Attack_Recall_Mean" in ordinary.columns:
            plt.figure(figsize=(10, 5.5))
            sns.barplot(data=ordinary, x="Protocol", y="Attack_Recall_Mean", hue="Model")
            plt.ylim(0, 1.02)
            plt.ylabel("Attack recall")
            plt.title("Attack recall across evaluation protocols")
            plt.xticks(rotation=15)
            plt.tight_layout()
            plt.savefig(figures / "attack_recall_by_protocol.png", dpi=300)
            plt.close()

        unseen = binary.loc[binary["Protocol"] == "unseen"].copy()
        if not unseen.empty and "Attack_Recall_Mean" in unseen.columns:
            pivot = unseen.pivot(index="Held_Out_Attack", columns="Model", values="Attack_Recall_Mean")
            pivot.to_csv(tables / "unseen_attack_recall.csv")
            plt.figure(figsize=(10, max(5, 0.45 * len(pivot))))
            sns.heatmap(pivot, annot=True, fmt=".3f", vmin=0, vmax=1, cmap="viridis")
            plt.title("Recall for attack families absent from training")
            plt.tight_layout()
            plt.savefig(figures / "unseen_attack_recall_heatmap.png", dpi=300)
            plt.close()

