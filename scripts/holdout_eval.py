#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
import sys

# allow running without installing the package
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from insolvenza.data import fetch_polish_companies_bankruptcy, train_test_split
from insolvenza.pipeline import build_model, build_preprocessor
from insolvenza.metrics import classification_report_from_scores
from sklearn.pipeline import Pipeline


def main() -> None:
    ds = fetch_polish_companies_bankruptcy()
    X, y = ds.X, ds.y
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=123)

    results = {}
    for name in ["logreg", "rf", "gb"]:
        key, clf = build_model(name)
        pre = build_preprocessor(X_tr)
        pipe = Pipeline(steps=[("preprocess", pre), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        scores_tr = pipe.predict_proba(X_tr)[:, 1]
        scores_te = pipe.predict_proba(X_te)[:, 1]
        rep_tr = classification_report_from_scores(y_tr.to_numpy(), scores_tr).to_dict()
        rep_te = classification_report_from_scores(y_te.to_numpy(), scores_te).to_dict()
        results[key] = {"train": rep_tr, "test": rep_te}

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
