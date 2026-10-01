"""Loading and validating the BattLeDIM 2018 files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import LEAKAGE_FILE, SCADA_FILE, SCADA_SHEETS

EXPECTED_LEAK_LOCATIONS = [
    "p31", "p158", "p183", "p232", "p257", "p369", "p427",
    "p461", "p538", "p628", "p654", "p673", "p810", "p866",
]
EXPECTED_SENSOR_COUNTS = {"pressures": 33, "demands": 82, "flows": 3, "levels": 1}


def load_leakages(path: Path | str = LEAKAGE_FILE) -> pd.DataFrame:
    """Load the leakage ground truth.

    The file uses ';' as column separator and ',' as decimal separator.
    With pandas defaults the header is read as ONE column and the read fails
    with a ParserError at line 2312, the first row containing a decimal comma.
    """
    df = pd.read_csv(path, sep=";", decimal=",", parse_dates=["Timestamp"])
    return df


def leak_columns(leaks: pd.DataFrame) -> list[str]:
    return [c for c in leaks.columns if c != "Timestamp"]


def load_scada(path: Path | str = SCADA_FILE, cache_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load the four SCADA sheets as a dict of DataFrames.

    Reading the 92 MB workbook takes ~1-2 minutes; if ``cache_dir`` is given,
    a pickle cache is written/read there (never committed to Git).
    """
    out: dict[str, pd.DataFrame] = {}
    for key, sheet in SCADA_SHEETS.items():
        cached = Path(cache_dir) / f"scada_{key}.pkl" if cache_dir else None
        if cached is not None and cached.exists():
            df = pd.read_pickle(cached)
        else:
            df = pd.read_excel(path, sheet_name=sheet)
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            if cached is not None:
                cached.parent.mkdir(parents=True, exist_ok=True)
                df.to_pickle(cached)
        out[key] = df
    return out


def validate_scada(scada: dict[str, pd.DataFrame]) -> dict:
    """Integrity checks. Returns a report dict; raises on hard failures."""
    base = scada["pressures"]["Timestamp"]
    report = {}
    for key, df in scada.items():
        ts = df["Timestamp"]
        n_sensors = df.shape[1] - 1
        report[key] = {
            "rows": len(df),
            "sensors": n_sensors,
            "duplicates": int(ts.duplicated().sum()),
            "missing_values": int(df.isna().sum().sum()),
            "irregular_steps": int((ts.diff().dropna() != pd.Timedelta("5min")).sum()),
            "aligned_with_pressures": bool((ts.values == base.values).all()) if len(ts) == len(base) else False,
        }
        if not report[key]["aligned_with_pressures"]:
            raise ValueError(f"Sheet '{key}' timestamps are not row-aligned with pressures")
    return report
