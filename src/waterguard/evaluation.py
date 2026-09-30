"""Metrics, including event-level (episode) analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def point_metrics(y_true, y_pred, y_prob=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
        out["pr_auc"] = average_precision_score(y_true, y_prob)
    return out


def episodes(timestamps: pd.Series, flag: pd.Series, min_gap: str = "0min") -> pd.DataFrame:
    """Contiguous runs of flag==1 (5-minute steps). Returns start, end, n_steps."""
    flag = flag.astype(int).to_numpy()
    ts = pd.Series(timestamps).reset_index(drop=True)
    starts, ends = [], []
    in_run = False
    for i, v in enumerate(flag):
        if v and not in_run:
            starts.append(i); in_run = True
        elif not v and in_run:
            ends.append(i - 1); in_run = False
    if in_run:
        ends.append(len(flag) - 1)
    rows = [{"start_idx": s, "end_idx": e, "start": ts[s], "end": ts[e], "n_steps": e - s + 1}
            for s, e in zip(starts, ends)]
    return pd.DataFrame(rows, columns=["start_idx", "end_idx", "start", "end", "n_steps"])


def event_level_summary(df: pd.DataFrame, min_steps: int = 12) -> pd.DataFrame:
    """For each severe episode of at least ``min_steps`` 5-minute steps (default 1 h),
    report whether the model raised any alert inside it and the delay to first alert.

    ``df`` needs Timestamp, target, prediction, risk_probability.
    """
    df = df.reset_index(drop=True)
    eps = episodes(df["Timestamp"], df["target"])
    eps = eps[eps["n_steps"] >= min_steps].copy()
    rows = []
    for _, e in eps.iterrows():
        seg = df.iloc[e.start_idx:e.end_idx + 1]
        alerts = seg[seg["prediction"] == 1]
        first = alerts["Timestamp"].min() if len(alerts) else pd.NaT
        rows.append({
            "start": e.start, "end": e.end,
            "duration_hours": e.n_steps * 5 / 60,
            "detected": bool(len(alerts)),
            "alerted_share": len(alerts) / len(seg),
            "hours_to_first_alert": (first - e.start).total_seconds() / 3600 if len(alerts) else np.nan,
            "max_risk": seg["risk_probability"].max(),
            "peak_total_leak": seg["total_leak"].max(),
        })
    return pd.DataFrame(rows)
