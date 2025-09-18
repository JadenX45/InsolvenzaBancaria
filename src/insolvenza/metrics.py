from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics as skm


@dataclass
class ClassificationReport:
    roc_auc: float
    pr_auc: float
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    threshold: float
    confusion_matrix: Tuple[int, int, int, int]  # tn, fp, fn, tp

    def to_dict(self) -> Dict[str, float]:
        tn, fp, fn, tp = self.confusion_matrix
        return {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "threshold": self.threshold,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }


def find_best_threshold(y_true: np.ndarray, y_scores: np.ndarray, beta: float = 2.0) -> float:
    """Select a decision threshold maximizing F-beta (defaults to recall-friendly F2).

    Works on probability scores. Returns threshold in [0,1].
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_score = -1.0
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        score = skm.fbeta_score(y_true, y_pred, beta=beta)
        if score > best_score:
            best_score, best_t = score, t
    return float(best_t)


def classification_report_from_scores(
    y_true: np.ndarray, y_scores: np.ndarray, threshold: float | None = None
) -> ClassificationReport:
    if threshold is None:
        threshold = find_best_threshold(y_true, y_scores)

    y_pred = (y_scores >= threshold).astype(int)
    roc_auc = skm.roc_auc_score(y_true, y_scores)
    pr_auc = skm.average_precision_score(y_true, y_scores)
    precision = skm.precision_score(y_true, y_pred, zero_division=0)
    recall = skm.recall_score(y_true, y_pred)
    f1 = skm.f1_score(y_true, y_pred)
    accuracy = skm.accuracy_score(y_true, y_pred)
    balanced_accuracy = skm.balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = skm.confusion_matrix(y_true, y_pred).ravel()
    return ClassificationReport(
        roc_auc=float(roc_auc),
        pr_auc=float(pr_auc),
        accuracy=float(accuracy),
        balanced_accuracy=float(balanced_accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        threshold=float(threshold),
        confusion_matrix=(int(tn), int(fp), int(fn), int(tp)),
    )

