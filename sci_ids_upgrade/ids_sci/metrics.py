from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        if high == 1.0:
            mask = (probabilities >= low) & (probabilities <= high)
        else:
            mask = (probabilities >= low) & (probabilities < high)
        if not np.any(mask):
            continue
        observed = float(np.mean(y_true[mask]))
        confidence = float(np.mean(probabilities[mask]))
        result += float(np.mean(mask)) * abs(observed - confidence)
    return result


def binary_metrics(y_true, y_pred, scores=None, probabilities=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / max(tn + fp, 1)
    result = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Attack_Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "FPR": fp / max(fp + tn, 1),
        "FNR": fn / max(fn + tp, 1),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    ranking_score = scores
    if ranking_score is None and probabilities is not None:
        ranking_score = probabilities[:, 1]
    if ranking_score is not None and np.unique(y_true).size == 2:
        result["ROC_AUC"] = roc_auc_score(y_true, ranking_score)
        result["PR_AUC"] = average_precision_score(y_true, ranking_score)
    else:
        result["ROC_AUC"] = np.nan
        result["PR_AUC"] = np.nan
    if probabilities is not None:
        positive = probabilities[:, 1]
        result["Brier"] = brier_score_loss(y_true, positive)
        result["ECE_10"] = expected_calibration_error(y_true, positive, bins=10)
    else:
        result["Brier"] = np.nan
        result["ECE_10"] = np.nan
    return result


def multiclass_metrics(y_true, y_pred, probabilities=None) -> dict:
    result = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Macro_Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro_Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro_F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Weighted_F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    result["Log_Loss"] = log_loss(y_true, probabilities) if probabilities is not None else np.nan
    return result

