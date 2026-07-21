from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
try:
    import xgboost
except ImportError:
    xgboost = None
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from .data import LoadedDataset
from .metrics import binary_metrics, multiclass_metrics
from .models import decision_scores, make_model, unwrap_tree_model
from .splits import (
    SplitResult,
    cap_stratified,
    group_split,
    random_split,
    temporal_split,
    unseen_attack_split,
)


def _append_csv(path: Path, rows: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path)
        pd.concat([existing, rows], ignore_index=True, sort=False).to_csv(path, index=False)
    else:
        rows.to_csv(path, index=False)


def _cap_unseen_test(
    frame: pd.DataFrame,
    held_out_attack: str,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    """Retain every rare held-out attack and use benign rows to fill the cap."""
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    attacks = frame.loc[frame["Label"] == held_out_attack]
    benign = frame.loc[frame["Label"] == "BENIGN"]
    if len(attacks) < max_rows:
        benign_slots = max_rows - len(attacks)
        benign_sample = benign.sample(n=min(benign_slots, len(benign)), random_state=seed)
        return pd.concat([benign_sample, attacks], ignore_index=True)

    attack_slots = max_rows // 2
    benign_slots = max_rows - attack_slots
    return pd.concat([
        benign.sample(n=min(benign_slots, len(benign)), random_state=seed),
        attacks.sample(n=attack_slots, random_state=seed),
    ], ignore_index=True)


def _save_explanation(model, X_test: pd.DataFrame, model_name: str, tag: str, out_dir: Path) -> None:
    tree = unwrap_tree_model(model)
    if hasattr(tree, "feature_importances_"):
        pd.DataFrame({
            "Feature": X_test.columns,
            "Importance": tree.feature_importances_,
        }).sort_values("Importance", ascending=False).to_csv(
            out_dir / f"feature_importance_{tag}.csv", index=False
        )
    if model_name != "xgboost":
        return
    try:
        import shap
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping SHAP: {exc}")
        return
    sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)
    explainer = shap.TreeExplainer(tree)
    values = explainer.shap_values(sample)
    array = np.asarray(values)
    if array.ndim == 3:
        array = np.mean(np.abs(array), axis=2)
    pd.DataFrame({
        "Feature": sample.columns,
        "Mean_Absolute_SHAP": np.mean(np.abs(array), axis=0),
    }).sort_values("Mean_Absolute_SHAP", ascending=False).to_csv(
        out_dir / f"shap_importance_{tag}.csv", index=False
    )
    shap.summary_plot(values, sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(out_dir / f"shap_summary_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close()


def _evaluate_one(
    loaded: LoadedDataset,
    split: SplitResult,
    protocol: str,
    model_name: str,
    seed: int,
    output_dir: Path,
    max_train_rows: int,
    max_test_rows: int,
    held_out_attack: str | None = None,
    explain: bool = False,
) -> tuple[dict, dict]:
    train_label = "Label" if held_out_attack is not None else "Binary_Label"
    train = cap_stratified(split.train, max_train_rows, train_label, seed)
    test = (
        _cap_unseen_test(split.test, held_out_attack, max_test_rows, seed + 1000)
        if held_out_attack is not None
        else cap_stratified(split.test, max_test_rows, "Binary_Label", seed + 1000)
    )
    if train["Binary_Label"].nunique() < 2 or test["Binary_Label"].nunique() < 2:
        raise ValueError(f"{protocol} split does not contain both binary classes")

    X_train = train[loaded.feature_columns]
    X_test = test[loaded.feature_columns]
    y_train = train["Binary_Label"].to_numpy()
    y_test = test["Binary_Label"].to_numpy()
    model = make_model(model_name, "binary", seed, y_train)

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_seconds = time.perf_counter() - start
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_seconds = time.perf_counter() - start
    probabilities, scores = decision_scores(model, X_test)
    row = binary_metrics(y_test, y_pred, scores=scores, probabilities=probabilities)
    run_id = f"{protocol}_{model_name}_{seed}" + (f"_{held_out_attack}" if held_out_attack else "")
    row.update({
        "Run_ID": run_id,
        "Task": "binary",
        "Protocol": protocol,
        "Model": model_name,
        "Seed": seed,
        "Held_Out_Attack": held_out_attack,
        "Training_Rows": len(train),
        "Testing_Rows": len(test),
        "Training_Attacks": int(y_train.sum()),
        "Testing_Attacks": int(y_test.sum()),
        "Feature_Count": len(loaded.feature_columns),
        "Training_Seconds": training_seconds,
        "Inference_Seconds": inference_seconds,
        "Flows_Per_Second": len(test) / max(inference_seconds, 1e-12),
    })

    detail_dir = output_dir / "details" / run_id
    detail_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(classification_report(
        y_test, y_pred, target_names=["Benign", "Attack"], output_dict=True, zero_division=0
    )).transpose().to_csv(detail_dir / "classification_report.csv")
    pd.DataFrame(confusion_matrix(y_test, y_pred, labels=[0, 1]),
                 index=["Actual_Benign", "Actual_Attack"],
                 columns=["Predicted_Benign", "Predicted_Attack"]).to_csv(
        detail_dir / "confusion_matrix.csv"
    )
    if explain:
        explain_dir = output_dir / "explanations"
        explain_dir.mkdir(parents=True, exist_ok=True)
        _save_explanation(model, X_test, model_name, run_id, explain_dir)
    return row, split.metadata


def run_experiments(
    loaded: LoadedDataset,
    output_dir: Path,
    protocols: list[str],
    models: list[str],
    seeds: list[int],
    max_train_rows: int,
    max_test_rows: int,
    explain: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_path = output_dir / "raw_metrics.csv"
    manifest_path = output_dir / "split_manifest.csv"
    existing = set()
    if metric_path.exists():
        existing = set(pd.read_csv(metric_path)["Run_ID"].astype(str))

    ordinary_protocols = [p for p in protocols if p != "unseen"]
    for seed in seeds:
        for protocol in ordinary_protocols:
            if protocol == "random":
                split = random_split(loaded.frame, "Binary_Label", seed)
            elif protocol == "group":
                split = group_split(loaded.frame, seed)
            elif protocol == "temporal":
                split = temporal_split(loaded.frame, novel_only=False)
            elif protocol == "temporal_novel":
                split = temporal_split(loaded.frame, novel_only=True)
            else:
                raise ValueError(f"Unknown protocol: {protocol}")
            for model_name in models:
                run_id = f"{protocol}_{model_name}_{seed}"
                if run_id in existing:
                    print(f"Skipping completed run {run_id}")
                    continue
                print(f"Running {run_id} ...")
                row, metadata = _evaluate_one(
                    loaded, split, protocol, model_name, seed, output_dir,
                    max_train_rows, max_test_rows,
                    explain=explain and seed == seeds[0],
                )
                _append_csv(metric_path, pd.DataFrame([row]))
                manifest = {"Run_ID": run_id, **metadata,
                            "Train_Rows_Before_Cap": len(split.train),
                            "Test_Rows_Before_Cap": len(split.test)}
                _append_csv(manifest_path, pd.DataFrame([manifest]))
                existing.add(run_id)

        if "unseen" in protocols:
            attacks = sorted(str(x) for x in loaded.frame["Label"].cat.categories if str(x) != "BENIGN")
            for attack in attacks:
                if int((loaded.frame["Label"] == attack).sum()) == 0:
                    continue
                split = unseen_attack_split(loaded.frame, attack, seed)
                for model_name in models:
                    run_id = f"unseen_{model_name}_{seed}_{attack}"
                    if run_id in existing:
                        continue
                    print(f"Running {run_id} ...")
                    row, metadata = _evaluate_one(
                        loaded, split, "unseen", model_name, seed, output_dir,
                        max_train_rows, max_test_rows, held_out_attack=attack,
                        explain=False,
                    )
                    _append_csv(metric_path, pd.DataFrame([row]))
                    manifest = {"Run_ID": run_id, **metadata,
                                "Train_Rows_Before_Cap": len(split.train),
                                "Test_Rows_Before_Cap": len(split.test)}
                    _append_csv(manifest_path, pd.DataFrame([manifest]))
                    existing.add(run_id)

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__ if xgboost is not None else None,
        "protocols": protocols,
        "models": models,
        "seeds": seeds,
        "max_train_rows": max_train_rows,
        "max_test_rows": max_test_rows,
        "feature_count": len(loaded.feature_columns),
        "source_files": loaded.source_files,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
