#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
import sys
import logging

import joblib

# allow running without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from insolvenza.data import fetch_polish_companies_bankruptcy
from insolvenza.pipeline import cross_validate, fit_full


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train bankruptcy prediction models on Polish dataset")
    p.add_argument("--model", default="logreg", help="Model to use: logreg|rf|gb")
    p.add_argument("--artifacts", default="artifacts", help="Output directory for artifacts")
    p.add_argument("--cv-folds", type=int, default=5)
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("train")
    args = parse_args()
    out_dir = Path(args.artifacts)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Fetching dataset …")
    ds = fetch_polish_companies_bankruptcy()
    X, y = ds.X, ds.y
    log.info("Dataset ready: X=%s, positives=%d (%.2f%%)", X.shape, int(y.sum()), float(100*y.mean()))

    log.info("Running cross-validation for model=%s folds=%d", args.model, args.cv_folds)
    cvres = cross_validate(X, y, model_name=args.model, n_splits=args.cv_folds)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    metrics_path = out_dir / f"metrics_{cvres.model_name}_{ts}.json"
    with open(metrics_path, "w") as f:
        json.dump(cvres.report.to_dict(), f, indent=2)

    log.info("Fitting final model on full data …")
    model = fit_full(X, y, model_name=args.model)
    model_path = out_dir / f"model_{cvres.model_name}_{ts}.joblib"
    joblib.dump(model, model_path)

    print("Saved:")
    print(" -", metrics_path)
    print(" -", model_path)
    print("Metrics:")
    print(json.dumps(cvres.report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
