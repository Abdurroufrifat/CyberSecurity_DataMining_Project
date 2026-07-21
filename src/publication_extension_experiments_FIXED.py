"""
Corrected publication-level experiments for CICIDS2017 IDS project.

Fixes compared with previous version:
1. Removes target-leakage columns from feature matrix, especially Multi_Label.
2. Saves classification reports and feature-importance tables as CSV files.
3. Produces clearer confusion matrices with proper class tick labels.
4. Runs three publishable experiments:
   - Binary random split XGBoost
   - Binary time-based split XGBoost (train Monday-Thursday, test Friday)
   - Multi-class random split XGBoost

"""

from __future__ import annotations

import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

RANDOM_STATE = 42
SAMPLE_SIZE = 300_000
TEST_SAMPLE_SIZE = 100_000


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_columns(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def load_all_raw_csv(dataset_dir: Path) -> pd.DataFrame:
    files = sorted([
        p for p in dataset_dir.glob("*.csv")
        if "combined" not in p.name.lower()
        and "preprocessed" not in p.name.lower()
        and "result" not in p.name.lower()
        and "summary" not in p.name.lower()
    ])
    if not files:
        raise FileNotFoundError(f"No raw CICIDS2017 CSV files found in {dataset_dir}")

    frames = []
    for path in files:
        print(f"Loading {path.name}...")
        df = pd.read_csv(path, low_memory=False, encoding="latin1")
        df = clean_columns(df)
        df["Source_File"] = path.name
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df = clean_dataframe(df)
    return df


def make_binary_label(label_series: pd.Series) -> pd.Series:
    return label_series.astype(str).str.strip().str.upper().apply(lambda x: 0 if x == "BENIGN" else 1)


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return numeric features while removing all label/leakage columns."""
    leakage_cols = [
        "Label",
        "Binary_Label",
        "Multi_Label",
        "Source_File",
        "Flow ID",
        "Source IP",
        "Destination IP",
        "Timestamp",
    ]
    X = df.drop(columns=[c for c in leakage_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=["number"]).copy()

    # Extra safeguard: remove any accidental label-like numeric column.
    bad_cols = [c for c in X.columns if "label" in c.lower() or c.lower() in {"class", "target", "y"}]
    if bad_cols:
        print("Removing possible leakage columns:", bad_cols)
        X = X.drop(columns=bad_cols, errors="ignore")

    return X


def save_confusion_matrix(y_true, y_pred, labels, out_path: Path, title: str):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_feature_importance(model, feature_names, out_png: Path, out_csv: Path, top_n: int = 20):
    importances = model.feature_importances_
    fi = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    fi = fi.sort_values("Importance", ascending=False)
    fi.to_csv(out_csv, index=False)

    top = fi.head(top_n).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["Feature"], top["Importance"])
    plt.xlabel("Feature Importance")
    plt.title("Top Feature Importance")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved feature importance: {out_png}")


def train_xgb_binary(X_train, X_test, y_train, y_test, out_dir: Path, tag: str):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Handle imbalance for binary XGBoost.
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    start = time.time()
    model.fit(X_train_s, y_train)
    train_time = time.time() - start
    y_pred = model.predict(X_test_s)

    results = {
        "Experiment": tag,
        "Model": "XGBoost",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0),
        "Training Time (s)": train_time,
        "Training Samples": len(y_train),
        "Testing Samples": len(y_test),
    }

    print(f"\n{tag} results")
    print(results)
    report = classification_report(y_test, y_pred, target_names=["Benign", "Attack"], output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(out_dir / f"classification_report_{tag}.csv")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Attack"], zero_division=0))

    save_confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
        out_path=out_dir / f"confusion_matrix_{tag}.png",
        title=f"Confusion Matrix - {tag}",
    )

    plot_feature_importance(
        model,
        X_train.columns,
        out_dir / f"feature_importance_{tag}.png",
        out_dir / f"feature_importance_{tag}.csv",
    )

    if tag == "random_split_binary_xgboost":
        optional_shap_analysis(model, scaler, X_train, out_dir)

    return pd.DataFrame([results])


def random_split_binary_experiment(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    print("\n=== Binary Random Split Experiment ===")
    df = df.copy()
    df["Binary_Label"] = make_binary_label(df["Label"])

    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    X = get_features(df)
    y = df["Binary_Label"]

    print("Feature count:", X.shape[1])
    print("Leakage check - any label-like columns:", [c for c in X.columns if "label" in c.lower()])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    return train_xgb_binary(X_train, X_test, y_train, y_test, out_dir, "random_split_binary_xgboost")


def time_based_binary_experiment(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    print("\n=== Binary Time-Based Split Experiment ===")
    df = df.copy()
    df["Binary_Label"] = make_binary_label(df["Label"])

    train_df = df[~df["Source_File"].str.contains("Friday", case=False, na=False)].copy()
    test_df = df[df["Source_File"].str.contains("Friday", case=False, na=False)].copy()

    if len(train_df) > SAMPLE_SIZE:
        train_df = train_df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
    if len(test_df) > TEST_SAMPLE_SIZE:
        test_df = test_df.sample(n=TEST_SAMPLE_SIZE, random_state=RANDOM_STATE)

    X_train = get_features(train_df)
    y_train = train_df["Binary_Label"]
    X_test = get_features(test_df)
    y_test = test_df["Binary_Label"]

    # Align columns in case train/test files differ after cleaning.
    common_cols = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    print("Train class distribution:", y_train.value_counts().to_dict())
    print("Test class distribution:", y_test.value_counts().to_dict())
    print("Feature count:", X_train.shape[1])
    print("Leakage check - any label-like columns:", [c for c in X_train.columns if "label" in c.lower()])

    return train_xgb_binary(X_train, X_test, y_train, y_test, out_dir, "time_based_binary_xgboost")


def multiclass_random_split_experiment(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    print("\n=== Multi-Class Random Split Experiment ===")
    df = df.copy()

    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    # Keep classes that have enough samples after sampling.
    counts = df["Label"].value_counts()
    valid = counts[counts >= 5].index
    df = df[df["Label"].isin(valid)].copy()

    le = LabelEncoder()
    y = le.fit_transform(df["Label"].astype(str))
    X = get_features(df)

    print("Number of classes:", len(le.classes_))
    print("Feature count:", X.shape[1])
    print("Leakage check - any label-like columns:", [c for c in X.columns if "label" in c.lower()])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    start = time.time()
    model.fit(X_train_s, y_train)
    train_time = time.time() - start
    y_pred = model.predict(X_test_s)

    results = pd.DataFrame([{
        "Experiment": "multiclass_random_split_xgboost",
        "Model": "XGBoost",
        "Accuracy": accuracy_score(y_test, y_pred),
        "Macro Precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "Macro Recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "Macro F1-score": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "Weighted F1-score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "Training Time (s)": train_time,
        "Training Samples": len(y_train),
        "Testing Samples": len(y_test),
        "Number of Classes": len(le.classes_),
    }])

    print(results.to_string(index=False))
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(out_dir / "classification_report_multiclass_random_split_xgboost.csv")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    # Save confusion matrix only if class count is not too large.
    save_confusion_matrix(
        y_test,
        y_pred,
        labels=list(range(len(le.classes_))),
        out_path=out_dir / "confusion_matrix_multiclass_random_split_xgboost.png",
        title="Confusion Matrix - Multi-class XGBoost",
    )

    pd.DataFrame({"Encoded_Label": range(len(le.classes_)), "Class_Name": le.classes_}).to_csv(
        out_dir / "multiclass_label_mapping.csv", index=False
    )

    plot_feature_importance(
        model,
        X_train.columns,
        out_dir / "feature_importance_multiclass_random_split_xgboost.png",
        out_dir / "feature_importance_multiclass_random_split_xgboost.csv",
    )

    results.to_csv(out_dir / "multiclass_xgboost_results.csv", index=False)
    return results


def optional_shap_analysis(model, scaler, X_train: pd.DataFrame, out_dir: Path):
    try:
        import shap
    except Exception:
        print("SHAP is not installed. To enable SHAP, run: pip install shap")
        return

    print("Running SHAP analysis on a small sample...")
    sample = X_train.sample(n=min(2000, len(X_train)), random_state=RANDOM_STATE)
    sample_s = scaler.transform(sample)
    sample_s_df = pd.DataFrame(sample_s, columns=sample.columns)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_s_df)

    plt.figure()
    shap.summary_plot(shap_values, sample_s_df, show=False, max_display=20)
    plt.tight_layout()
    out_path = out_dir / "shap_summary_xgboost_fixed.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary: {out_path}")


def main():
    root = Path(__file__).resolve().parent
    project_root = root if (root / "dataset").exists() else root.parent
    dataset_dir = project_root / "dataset"
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}. Make sure the project root contains the dataset folder."
        )
    out_dir = project_root / "publication_results_fixed"
    out_dir.mkdir(exist_ok=True)

    df = load_all_raw_csv(dataset_dir)
    print("Cleaned full data shape:", df.shape)
    print("Class distribution:")
    print(df["Label"].value_counts())

    results = []
    results.append(random_split_binary_experiment(df, out_dir))
    results.append(time_based_binary_experiment(df, out_dir))
    results.append(multiclass_random_split_experiment(df, out_dir))

    final_results = pd.concat(results, ignore_index=True, sort=False)
    final_results.to_csv(out_dir / "publication_experiment_summary_fixed.csv", index=False)

    print("\nAll corrected publication experiments completed.")
    print(final_results.to_string(index=False))
    print(f"Results saved in: {out_dir}")


if __name__ == "__main__":
    main()
