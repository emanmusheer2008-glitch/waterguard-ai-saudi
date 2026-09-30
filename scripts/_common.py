"""Shared helpers for scripts: make src/ importable and build the V2 dataset."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waterguard.config import DATA_DIR  # noqa: E402
from waterguard.data import load_leakages, load_scada, validate_scada  # noqa: E402
from waterguard.features import build_features_v2, feature_columns  # noqa: E402
from waterguard.target import add_target, chronological_split  # noqa: E402

CACHE = DATA_DIR / "cache"


def load_v2_dataset(threshold=None):
    scada = load_scada(cache_dir=CACHE)
    report = validate_scada(scada)
    X = build_features_v2(scada)
    leaks = load_leakages()
    tgt = add_target(leaks) if threshold is None else add_target(leaks, threshold)
    df = X.merge(tgt, on="Timestamp").sort_values("Timestamp").reset_index(drop=True)
    return df, feature_columns(X), leaks, report
