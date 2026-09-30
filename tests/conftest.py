import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waterguard.config import LEAKAGE_FILE, MODELS_DIR, SCADA_FILE  # noqa: E402

needs_leakage = pytest.mark.skipif(not LEAKAGE_FILE.exists(), reason="data/2018_Leakages.csv not present (see README: Data setup)")
needs_scada = pytest.mark.skipif(not SCADA_FILE.exists(), reason="data/2018_SCADA.xlsx not present (see README: Data setup)")
needs_models = pytest.mark.skipif(not (MODELS_DIR / "waterguard_rf_v2.joblib").exists(), reason="model artifacts not present (run src/train_v2.py)")


@pytest.fixture
def synthetic_scada():
    """Small synthetic SCADA dict with the real BattLeDIM column layout."""
    rng = np.random.default_rng(0)
    n = 50
    ts = pd.date_range("2018-01-01", periods=n, freq="5min")
    def sheet(names, loc):
        df = pd.DataFrame(rng.normal(loc, 1, size=(n, len(names))), columns=names)
        df.insert(0, "Timestamp", ts)
        return df
    return {
        "pressures": sheet([f"n{i}" for i in range(33)], 50),
        "demands": sheet([f"n{i}" for i in range(100, 182)], 200),
        "flows": sheet(["p227", "p235", "PUMP_1"], 100),
        "levels": sheet(["T1"], 3),
    }
