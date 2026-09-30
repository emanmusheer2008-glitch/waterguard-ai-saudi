"""Target definition and chronological split."""
from __future__ import annotations

import pandas as pd

from .config import SEVERE_THRESHOLD, TRAIN_FRAC, VAL_FRAC
from .data import leak_columns


def add_target(leaks: pd.DataFrame, threshold: float = SEVERE_THRESHOLD) -> pd.DataFrame:
    """Return Timestamp, total_leak, target.

    target = 1 when total benchmark leakage >= ``threshold``
    (experimental benchmark severity threshold, not an official standard).
    """
    out = leaks[["Timestamp"]].copy()
    out["total_leak"] = leaks[leak_columns(leaks)].sum(axis=1)
    out["target"] = (out["total_leak"] >= threshold).astype(int)
    return out


def chronological_split(df: pd.DataFrame, train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC):
    """Split a time-sorted frame into train / validation / test by position (no shuffling)."""
    df = df.sort_values("Timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
