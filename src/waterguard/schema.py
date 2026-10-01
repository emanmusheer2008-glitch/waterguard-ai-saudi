"""Input schema for new data: one wide table with a Timestamp column and prefixed sensor columns.

Prefixes are needed because pressure and demand sensors share some node IDs
(e.g. n1 has both a pressure sensor and a demand meter in L-Town).

    Timestamp | P_n1 ... P_n769 (33) | D_n1 ... (82) | F_p227 F_p235 F_PUMP_1 | L_T1

This module only reshapes data; validation lives in waterguard.service.
"""
from __future__ import annotations

import pandas as pd

GROUP_PREFIX = {"pressures": "P_", "demands": "D_", "flows": "F_", "levels": "L_"}
GROUP_UNITS = {"pressures": "m (pressure head)", "demands": "L/h", "flows": "m³/h", "levels": "m"}
# BattLeDIM workbook sheet names (an uploaded .xlsx may use this 4-sheet layout instead of one wide sheet)
SHEET_NAMES = {"pressures": "Pressures (m)", "demands": "Demands (L_h)", "flows": "Flows (m3_h)", "levels": "Levels (m)"}


def scada_to_wide(scada: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """BattLeDIM-style dict of sheets -> one wide, prefixed table (rows aligned by position)."""
    out = scada["pressures"][["Timestamp"]].reset_index(drop=True).copy()
    for group, prefix in GROUP_PREFIX.items():
        part = scada[group].drop(columns="Timestamp").reset_index(drop=True)
        out = pd.concat([out, part.add_prefix(prefix)], axis=1)
    return out


def wide_to_scada(wide: pd.DataFrame, schema: dict[str, list[str]]) -> dict[str, pd.DataFrame]:
    """Wide prefixed table -> dict of sheets in the exact column order the model was trained with."""
    out = {}
    for group, prefix in GROUP_PREFIX.items():
        cols = [prefix + c for c in schema[group]]
        part = wide[["Timestamp"] + cols].copy()
        part.columns = ["Timestamp"] + list(schema[group])
        out[group] = part.reset_index(drop=True)
    return out


def expected_columns(schema: dict[str, list[str]]) -> list[str]:
    return ["Timestamp"] + [GROUP_PREFIX[g] + c for g in GROUP_PREFIX for c in schema[g]]
