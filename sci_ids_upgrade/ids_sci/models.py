from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
try:
    from xgboost import XGBClassifier
except ImportError:  # The other baselines remain usable during environment setup.
    XGBClassifier = None


MODEL_NAMES = ["logistic_regression", "decision_tree", "random_forest", "linear_svm", "xgboost"]


def make_model(name: str, task: str, seed: int, y_train: np.ndarray) -> Any:
    if name == "logistic_regression":
        return Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(
                solver="saga",
                max_iter=500,
                class_weight="balanced",
                random_state=seed,
            )),
        ])
    if name == "linear_svm":
        return Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LinearSVC(
                class_weight="balanced",
                random_state=seed,
                max_iter=10_000,
            )),
        ])
    if name == "decision_tree":
        return DecisionTreeClassifier(
            class_weight="balanced",
            min_samples_leaf=2,
            random_state=seed,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("XGBoost is not installed. Run: python -m pip install -r requirements.txt")
        common = dict(
            n_estimators=300,
            learning_rate=0.08,
            max_depth=6,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
        if task == "binary":
            neg = int((y_train == 0).sum())
            pos = int((y_train == 1).sum())
            return XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=neg / max(pos, 1),
                **common,
            )
        classes = int(np.unique(y_train).size)
        return XGBClassifier(
            objective="multi:softprob",
            num_class=classes,
            eval_metric="mlogloss",
            **common,
        )
    raise ValueError(f"Unknown model: {name}")


def decision_scores(model: Any, X) -> tuple[np.ndarray | None, np.ndarray | None]:
    probabilities = None
    scores = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            scores = probabilities[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
    return probabilities, scores


def unwrap_tree_model(model: Any) -> Any:
    if isinstance(model, Pipeline):
        return model.named_steps.get("classifier")
    return model
