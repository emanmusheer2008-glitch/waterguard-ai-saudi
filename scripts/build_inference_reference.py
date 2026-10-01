"""Build the small reference files that the inference service needs at run time.

Everything here is derived from the TRAINING period only (1 Jan - 7 Aug 2018),
so applying it to new data never uses validation/test information.
Does NOT retrain or modify the V2 model.

Outputs (committed, in models/):
  inference_reference_v2.json   input schema, training value ranges per input column,
                                explanation signals + their typical (non-severe training) values
  pressure_typical_median_v2.csv  per 5-min slot of day x pressure sensor: usual value
  pressure_typical_spread_v2.csv  per 5-min slot of day x pressure sensor: spread (std, floored)
"""
import json

import pandas as pd

from _common import CACHE, load_v2_dataset
from waterguard.config import MODELS_DIR, OUTPUTS_DIR, SEVERE_THRESHOLD, V2_ALERT_THRESHOLD
from waterguard.data import load_scada
from waterguard.inspection import STD_FLOOR_M, fit_typical_pressure
from waterguard.schema import GROUP_PREFIX, scada_to_wide
from waterguard.target import chronological_split

N_EXPLAIN, N_PERM = 12, 8  # identical rule to scripts/build_dashboard_data.py

df, feats, _, _ = load_v2_dataset()
train, _, _ = chronological_split(df)
scada = load_scada(cache_dir=CACHE)

# ---------------- input schema (BattLeDIM column order, prefixed to avoid name clashes)
schema = {g: [c for c in scada[g].columns if c != "Timestamp"] for g in GROUP_PREFIX}
wide = scada_to_wide(scada)
wide_train = wide[wide["Timestamp"].isin(train["Timestamp"])]
cols = [c for c in wide.columns if c != "Timestamp"]
ranges = {c: {"min": float(wide_train[c].min()), "max": float(wide_train[c].max()),
              "q01": float(wide_train[c].quantile(0.01)), "q99": float(wide_train[c].quantile(0.99))}
          for c in cols}

# ---------------- explanation signals: same selection rule as the benchmark dashboard
imp = pd.read_csv(OUTPUTS_DIR / "feature_importance_v2.csv")
perm = pd.read_csv(OUTPUTS_DIR / "permutation_importance_v2.csv").sort_values("validation_auc_drop_mean", ascending=False)
signals = list(dict.fromkeys(imp["feature"].head(N_EXPLAIN).tolist() + perm["feature"].head(N_PERM).tolist()))
normal = train[train["target"] == 0]
typical = {f: {"median": float(normal[f].median()), "q05": float(normal[f].quantile(0.05)),
               "q95": float(normal[f].quantile(0.95)), "std": float(normal[f].std())} for f in signals}

ref = {
    "model_version": "WaterGuard V2 (Random Forest, 60 features)",
    "training_period": {"start": str(train["Timestamp"].min()), "end": str(train["Timestamp"].max()),
                        "rows": int(len(train))},
    "alert_threshold": V2_ALERT_THRESHOLD,
    "severe_threshold_m3h": SEVERE_THRESHOLD,
    "sampling_interval_minutes": 5,
    "schema": schema,
    "prefixes": GROUP_PREFIX,
    "feature_order": feats,
    "input_ranges_train": ranges,
    "explain_signals": signals,
    "explain_typical": typical,
    "pressure_std_floor_m": STD_FLOOR_M,
}
(MODELS_DIR / "inference_reference_v2.json").write_text(json.dumps(ref, indent=1))

med, spread = fit_typical_pressure(scada["pressures"], train.loc[train["target"] == 0, "Timestamp"])
med.to_csv(MODELS_DIR / "pressure_typical_median_v2.csv", index_label="slot", float_format="%.10g")
spread.to_csv(MODELS_DIR / "pressure_typical_spread_v2.csv", index_label="slot", float_format="%.10g")
print(f"schema: {', '.join(f'{g}={len(v)}' for g, v in schema.items())}; {len(cols)} input columns; "
      f"{len(signals)} explanation signals")
print("Done.")
