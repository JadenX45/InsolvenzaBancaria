from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import logging
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .metrics import classification_report_from_scores, ClassificationReport

log = logging.getLogger(__name__)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = list(X.columns)
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, numeric_features)], remainder="drop"
    )
    return preprocessor


def build_model(name: str) -> Tuple[str, object]:
    name = name.lower()
    if name in {"logreg", "logistic"}:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None)
        return "logreg", clf
    if name in {"rf", "random_forest", "random-forest"}:
        clf = RandomForestClassifier(
            n_estimators=200, max_depth=None, n_jobs=-1, class_weight="balanced_subsample", random_state=42
        )
        return "rf", clf
    if name in {"gb", "gboost", "gradient_boosting"}:
        clf = GradientBoostingClassifier(random_state=42)
        return "gb", clf
    raise ValueError(f"Unknown model name: {name}")


@dataclass
class CVResult:
    model_name: str
    report: ClassificationReport
    y_true: np.ndarray
    y_scores: np.ndarray


def cross_validate(X: pd.DataFrame, y: pd.Series, model_name: str = "logreg", n_splits: int = 5) -> CVResult:
    model_key, clf = build_model(model_name)
    pre = build_preprocessor(X)
    pipe = Pipeline(steps=[("preprocess", pre), ("clf", clf)])

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # probability scores via cross_val_predict; for GB and RF proba exists; for LogReg as well
    log.info(
        "Starting %d-fold CV for %s on X=%s; positives=%d (%.2f%%)",
        n_splits,
        model_key,
        X.shape,
        int(y.sum()),
        float(100 * y.mean()),
    )
    t0 = time.perf_counter()
    y_scores = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    log.info("CV finished for %s in %.1fs", model_key, time.perf_counter() - t0)
    y_true = y.to_numpy()
    report = classification_report_from_scores(y_true, y_scores)
    return CVResult(model_name=model_key, report=report, y_true=y_true, y_scores=y_scores)


def fit_full(X: pd.DataFrame, y: pd.Series, model_name: str = "logreg") -> Pipeline:
    model_key, clf = build_model(model_name)
    pre = build_preprocessor(X)
    pipe = Pipeline(steps=[("preprocess", pre), ("clf", clf)])
    pipe.fit(X, y)
    return pipe
