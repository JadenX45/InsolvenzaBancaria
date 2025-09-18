from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import pandas as pd


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    metadata: dict


def fetch_polish_companies_bankruptcy() -> Dataset:
    """Fetch the Polish Companies Bankruptcy dataset from UCI.

    Returns a Dataset object with features X, target y ('class' 0/1), and metadata.
    Requires the `ucimlrepo` package.
    """
    from ucimlrepo import fetch_ucirepo  # imported here to avoid hard dependency if unused

    ds = fetch_ucirepo(id=365)
    X: pd.DataFrame = ds.data.features.copy()
    y: pd.Series = ds.data.targets.iloc[:, 0].astype(int).rename("class")
    return Dataset(X=X, y=y, metadata=ds.metadata)


def train_test_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    from sklearn.model_selection import train_test_split as sk_split

    return sk_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

