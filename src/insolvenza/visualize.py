from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics as skm
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_roc_pr(y_true: np.ndarray, y_scores: np.ndarray, out_path_prefix: Path) -> None:
    fpr, tpr, _ = skm.roc_curve(y_true, y_scores)
    roc_auc = skm.roc_auc_score(y_true, y_scores)
    precision, recall, _ = skm.precision_recall_curve(y_true, y_scores)
    pr_auc = skm.average_precision_score(y_true, y_scores)

    # ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    path = out_path_prefix.with_suffix(".roc.png")
    _ensure_dir(path)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR AUC={pr_auc:.3f}")
    base = float(np.mean(y_true))
    ax.hlines(base, 0, 1, colors="gray", linestyles="--", label=f"Baseline={base:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    path = out_path_prefix.with_suffix(".pr.png")
    _ensure_dir(path)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
    out_path: Path,
    normalize: str | None = None,
) -> None:
    """Plot confusion matrix.

    normalize: one of None, 'true', 'pred', 'all'. If provided, percentages are shown.
    """
    y_pred = (y_scores >= threshold).astype(int)
    if normalize in {"true", "pred", "all"}:
        cm = skm.confusion_matrix(y_true, y_pred, normalize=normalize)
        # show as percentage
        data = cm * 100.0
        fmt = ".1f"
        title_extra = f" (% {normalize})"
        cbar = True
    else:
        cm = skm.confusion_matrix(y_true, y_pred)
        data = cm
        fmt = "d"
        title_extra = ""
        cbar = False

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", cbar=cbar, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix{title_extra} (threshold={threshold:.2f})")
    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_calibration(y_true: np.ndarray, y_scores: np.ndarray, out_path: Path, n_bins: int = 10) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=n_bins, strategy="uniform")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted value")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration plot")
    ax.legend(loc="upper left")
    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_score_hist(y_true: np.ndarray, y_scores: np.ndarray, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(y_scores[y_true == 0], bins=30, color="#4e79a7", alpha=0.6, label="Class 0", ax=ax)
    sns.histplot(y_scores[y_true == 1], bins=30, color="#e15759", alpha=0.6, label="Class 1", ax=ax)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Score distribution by class")
    ax.legend()
    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def get_feature_importances(pipe: Pipeline, topn: int = 15) -> pd.DataFrame:
    preprocess = pipe.named_steps["preprocess"]
    clf = pipe.named_steps["clf"]
    feature_names = list(preprocess.get_feature_names_out())
    # remove ColumnTransformer prefix "num__"
    feature_names = [n.split("__", 1)[-1] for n in feature_names]

    importances = None
    if hasattr(clf, "feature_importances_"):
        importances = np.asarray(clf.feature_importances_)
    elif hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_)
        if coef.ndim == 2:
            coef = coef[0]
        importances = np.abs(coef)
    else:
        return pd.DataFrame(columns=["feature", "importance"]).set_index("feature")

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(topn)
    return df


def plot_feature_importances(pipe: Pipeline, out_path: Path, topn: int = 15) -> None:
    df = get_feature_importances(pipe, topn=topn)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(df))))
    sns.barplot(data=df, y="feature", x="importance", ax=ax, color="#4e79a7")
    ax.set_title("Top feature importances")
    ax.set_xlabel("Importance (abs coef or impurity)")
    ax.set_ylabel("")
    _ensure_dir(out_path)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
