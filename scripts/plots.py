#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import logging

# allow running without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from insolvenza.data import fetch_polish_companies_bankruptcy
from insolvenza.pipeline import cross_validate, fit_full
from insolvenza.visualize import (
    plot_roc_pr,
    plot_confusion,
    plot_calibration,
    plot_score_hist,
    plot_feature_importances,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate evaluation plots")
    p.add_argument("--model", default=None, help="logreg|rf|gb (deprecated in favor of --models)")
    p.add_argument("--models", default="logreg,rf,gb", help="Comma-separated models to evaluate")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--outdir", default="artifacts/plots")
    p.add_argument("--topn", type=int, default=15, help="Top features to show")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("plots")
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log.info("Fetching dataset …")
    ds = fetch_polish_companies_bankruptcy()
    X, y = ds.X, ds.y
    models = [m.strip() for m in (args.models or (args.model or "logreg")).split(",") if m.strip()]
    for model in models:
        log.info("Running CV to get out-of-fold scores for %s …", model)
        cv = cross_validate(X, y, model_name=model, n_splits=args.folds)

        prefix = outdir / f"{cv.model_name}"
        plot_roc_pr(cv.y_true, cv.y_scores, prefix)
        # counts
        plot_confusion(cv.y_true, cv.y_scores, cv.report.threshold, outdir / f"{cv.model_name}.cm.png")
        # percentages by true class (row-wise)
        plot_confusion(
            cv.y_true,
            cv.y_scores,
            cv.report.threshold,
            outdir / f"{cv.model_name}.cm_row_pct.png",
            normalize="true",
        )
        plot_calibration(cv.y_true, cv.y_scores, outdir / f"{cv.model_name}.calibration.png")
        plot_score_hist(cv.y_true, cv.y_scores, outdir / f"{cv.model_name}.score_hist.png")

        log.info("Fitting full model for feature importances for %s …", model)
        pipe = fit_full(X, y, model_name=model)
        plot_feature_importances(pipe, outdir / f"{cv.model_name}.features.png", topn=args.topn)

    log.info("Plots saved under %s", outdir)


if __name__ == "__main__":
    main()
