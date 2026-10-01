"""Loaders for the dashboard. Only reads small files from outputs/ so the app
can be deployed without the raw dataset or the 25 MB model files."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import OUTPUTS_DIR

REQUIRED_OUTPUTS = [
    "predictions_v2.csv",
    "predictions_v1.csv",
    "feature_importance_v2.csv",
    "feature_importance_v1.csv",
    "threshold_validation_v2.csv",
    "dashboard_context_v2.csv.gz",
    "typical_conditions_v2.csv",
    "curves_v2.csv",
    "event_summary_v2.csv",
    "audit_report.json",
    "pressure_deviation_v2.csv.gz",
    "network_links.csv",
    "network_sensors.csv",
    "inspection_check_v2.csv",
    "permutation_importance_v2.csv",
    "experiments/baseline_comparison.csv",
    "experiments/severity_threshold_sensitivity.csv",
]


def missing_outputs(outputs_dir: Path = OUTPUTS_DIR) -> list[str]:
    return [f for f in REQUIRED_OUTPUTS if not (outputs_dir / f).exists()]


def load_all(outputs_dir: Path = OUTPUTS_DIR) -> dict:
    o = outputs_dir
    return {
        "pred": pd.read_csv(o / "predictions_v2.csv", parse_dates=["Timestamp"]),
        "pred_v1": pd.read_csv(o / "predictions_v1.csv", parse_dates=["Timestamp"]),
        "imp": pd.read_csv(o / "feature_importance_v2.csv"),
        "imp_v1": pd.read_csv(o / "feature_importance_v1.csv"),
        "perm": pd.read_csv(o / "permutation_importance_v2.csv"),
        "thr_val": pd.read_csv(o / "threshold_validation_v2.csv"),
        "ctx": pd.read_csv(o / "dashboard_context_v2.csv.gz", parse_dates=["Timestamp"]),
        "typical": pd.read_csv(o / "typical_conditions_v2.csv"),
        "curves": pd.read_csv(o / "curves_v2.csv"),
        "events": pd.read_csv(o / "event_summary_v2.csv", parse_dates=["start", "end"]),
        "audit": json.loads((o / "audit_report.json").read_text()),
        "pdev": pd.read_csv(o / "pressure_deviation_v2.csv.gz", parse_dates=["Timestamp"]),
        "links": pd.read_csv(o / "network_links.csv"),
        "sensors": pd.read_csv(o / "network_sensors.csv"),
        "inspect_check": pd.read_csv(o / "inspection_check_v2.csv", parse_dates=["episode_start", "episode_end"]),
        "baselines": pd.read_csv(o / "experiments" / "baseline_comparison.csv"),
        "severity": pd.read_csv(o / "experiments" / "severity_threshold_sensitivity.csv"),
    }
