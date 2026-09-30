"""Feature engineering for V1 and V2.

Both builders replicate the original scripts exactly (column order
included) so saved models can be re-applied. All features use only the
current and *previous* timestamps (``diff``), never future values, and
never the leakage ground truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUND_TRUTH_PREFIXES = ("total_leak", "target", "severe", "leak")


def _parts(scada):
    P = scada["pressures"].drop(columns="Timestamp")
    D = scada["demands"].drop(columns="Timestamp")
    F = scada["flows"].drop(columns="Timestamp")
    L = scada["levels"].drop(columns="Timestamp")
    return P, D, F, L


def build_features_v1(scada: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """20 aggregate features (includes calendar month)."""
    P, D, F, L = _parts(scada)
    X = pd.DataFrame({"Timestamp": scada["pressures"]["Timestamp"]})
    X["pressure_mean"] = P.mean(axis=1)
    X["pressure_min"] = P.min(axis=1)
    X["pressure_max"] = P.max(axis=1)
    X["pressure_std"] = P.std(axis=1)
    X["pressure_range"] = P.max(axis=1) - P.min(axis=1)
    X["demand_total"] = D.sum(axis=1)
    X["demand_mean"] = D.mean(axis=1)
    X["demand_std"] = D.std(axis=1)
    X["flow_total"] = F.sum(axis=1)
    X["flow_mean"] = F.mean(axis=1)
    X["flow_std"] = F.std(axis=1)
    X["tank_level"] = L.iloc[:, 0]
    X["hour"] = X["Timestamp"].dt.hour
    X["dayofweek"] = X["Timestamp"].dt.dayofweek
    X["month"] = X["Timestamp"].dt.month
    for col in ["pressure_mean", "pressure_min", "demand_total", "flow_total", "tank_level"]:
        X[f"{col}_change_5m"] = X[col].diff()
    return X


def build_features_v2(scada: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """60 hydraulic features: 33 pressures, summaries, flows, tank, cyclical hour, changes. No month."""
    P, D, F, L = _parts(scada)
    X = pd.DataFrame({"Timestamp": scada["pressures"]["Timestamp"]})
    for col in P.columns:
        X[f"P_{col}"] = P[col]
    X["pressure_mean"] = P.mean(axis=1)
    X["pressure_min"] = P.min(axis=1)
    X["pressure_max"] = P.max(axis=1)
    X["pressure_std"] = P.std(axis=1)
    X["pressure_range"] = P.max(axis=1) - P.min(axis=1)
    X["demand_total"] = D.sum(axis=1)
    X["demand_mean"] = D.mean(axis=1)
    X["demand_std"] = D.std(axis=1)
    for col in F.columns:
        X[f"F_{col}"] = F[col]
    X["flow_total"] = F.sum(axis=1)
    X["flow_mean"] = F.mean(axis=1)
    X["flow_std"] = F.std(axis=1)
    X["tank_level"] = L.iloc[:, 0]
    hour = X["Timestamp"].dt.hour + X["Timestamp"].dt.minute / 60
    X["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    for col in ["pressure_mean", "pressure_min", "pressure_std", "flow_total", "tank_level"]:
        X[f"{col}_diff_5m"] = X[col].diff()
        X[f"{col}_diff_30m"] = X[col].diff(6)
    return X


def feature_columns(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c != "Timestamp"]


def feature_category(name: str) -> str:
    """Human-readable hydraulic category for a feature name."""
    if name.startswith("P_"):
        return "Individual pressure sensor"
    if name.startswith("F_"):
        return "Individual flow / pump"
    if "diff" in name or "change" in name:
        return "Short-term change"
    if name.startswith("pressure"):
        return "Network pressure summary"
    if name.startswith("flow"):
        return "Network flow summary"
    if name.startswith("demand"):
        return "Demand summary"
    if name.startswith("tank"):
        return "Tank level"
    if name.startswith("hour") or name in {"dayofweek", "month"}:
        return "Time of day / calendar"
    return "Other"
