"""Central configuration: paths and experiment constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

SCADA_FILE = DATA_DIR / "2018_SCADA.xlsx"
LEAKAGE_FILE = DATA_DIR / "2018_Leakages.csv"
NETWORK_FILE = DATA_DIR / "L-TOWN.inp"

SCADA_SHEETS = {
    "pressures": "Pressures (m)",
    "demands": "Demands (L_h)",
    "flows": "Flows (m3_h)",
    "levels": "Levels (m)",
}

# Experimental benchmark severity threshold (NOT an official engineering,
# utility, regulatory or Saudi threshold). Total leakage across the 14
# BattLeDIM leak locations at a 5-minute timestamp, in m3/h (unit stated in
# the BattLeDIM README on Zenodo).
SEVERE_THRESHOLD = 40.0

# Chronological split fractions used by V2 (60 / 10 / 30).
TRAIN_FRAC = 0.60
VAL_FRAC = 0.10  # validation = next 10%; test = remaining 30%

# V2 alert threshold selected on validation data only (stored in
# models/threshold_v2.joblib as 0.22000000000000003 from np.arange).
V2_ALERT_THRESHOLD = 0.22

# Verified V2 test results (from outputs/predictions_v2.csv).
V2_VERIFIED = {
    "tn": 22332, "fp": 210, "fn": 5561, "tp": 3433,
    "precision": 0.9424, "recall": 0.3817, "f1": 0.5433, "roc_auc": 0.9501,
}
V1_VERIFIED = {"precision": 0.4255, "recall": 0.0022, "f1": 0.0044, "roc_auc": 0.8602}

RANDOM_STATE = 42
