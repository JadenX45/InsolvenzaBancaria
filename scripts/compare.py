#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
import sys
import logging
import argparse

# allow running without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from insolvenza.data import fetch_polish_companies_bankruptcy
from insolvenza.pipeline import cross_validate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare models with cross-validation")
    p.add_argument(
        "--models",
        default="logreg,rf,gb",
        help="Comma-separated list among: logreg, rf, gb",
    )
    p.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("compare")

    log.info("Fetching UCI Polish Companies Bankruptcy dataset …")
    ds = fetch_polish_companies_bankruptcy()
    X, y = ds.X, ds.y
    pos_rate = float(y.mean())
    log.info("Dataset ready: X=%s, y=%s, positive rate=%.3f", X.shape, y.shape, pos_rate)

    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    log.info("Models to evaluate: %s (folds=%d)", ", ".join(models), args.folds)

    results = []
    for m in models:
        log.info("Evaluating model: %s", m)
        cv = cross_validate(X, y, model_name=m, n_splits=args.folds)
        d = cv.report.to_dict()
        d["model"] = m
        results.append(d)
        log.info(
            "Done %s | ROC-AUC=%.3f PR-AUC=%.3f Recall=%.3f F1=%.3f",
            m,
            d["roc_auc"],
            d["pr_auc"],
            d["recall"],
            d["f1"],
        )

    df = pd.DataFrame(results)
    out = ROOT / "artifacts" / "model_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log.info("Saved comparison to %s", out)


if __name__ == "__main__":
    main()
