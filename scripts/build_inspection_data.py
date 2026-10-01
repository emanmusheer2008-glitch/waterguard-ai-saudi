"""Build the supplementary inspection-guidance and held-out importance files.

Does NOT retrain or change V2. Needs data/2018_SCADA.xlsx, data/2018_Leakages.csv,
data/L-TOWN.inp (python scripts/download_data.py) and the saved V2 model.

Outputs (all small, committed):
  outputs/pressure_deviation_v2.csv.gz  per-test-timestamp deviation of each pressure sensor
                                        from its usual level at that time of day
  outputs/network_links.csv             L-Town pipes/pumps/valves as line segments (map)
  outputs/network_sensors.csv           pressure-sensor coordinates (map)
  outputs/inspection_check_v2.csv       evaluation only: rank of the sensor nearest the true
                                        leaking pipe, per severe test episode
  outputs/permutation_importance_v2.csv held-out (validation and test) permutation importance
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from _common import CACHE, load_v2_dataset
from waterguard.config import MODELS_DIR, NETWORK_FILE, OUTPUTS_DIR, RANDOM_STATE
from waterguard.data import load_scada
from waterguard.evaluation import episodes
from waterguard.inspection import (fit_typical_pressure, link_segments, parse_inp, pressure_deviation,
                                   sensor_rank_check)
from waterguard.target import chronological_split

INP = NETWORK_FILE
if not INP.exists():
    raise SystemExit("data/L-TOWN.inp missing - run python scripts/download_data.py")

df, feats, leaks, _ = load_v2_dataset()
train, val, test = chronological_split(df)
pressures = load_scada(cache_dir=CACHE)["pressures"]

# ---------------- pressure deviation (typical = non-severe TRAINING rows only)
med, spread = fit_typical_pressure(pressures, train.loc[train["target"] == 0, "Timestamp"])
dev = pressure_deviation(pressures, test["Timestamp"], med, spread)
sensor_cols = [c for c in dev.columns if c != "Timestamp"]
dev[sensor_cols] = dev[sensor_cols].round(2)
dev.to_csv(OUTPUTS_DIR / "pressure_deviation_v2.csv.gz", index=False, compression="gzip")
print("pressure deviation:", dev.shape)

# ---------------- network layout
nodes, links = parse_inp(INP)
seg = link_segments(nodes, links)
seg.round(2).to_csv(OUTPUTS_DIR / "network_links.csv", index=False)
sensors = nodes[nodes["id"].isin(sensor_cols)].set_index("id")
assert len(sensors) == len(sensor_cols), "some pressure sensors have no coordinates"
sensors.round(2).reset_index().to_csv(OUTPUTS_DIR / "network_sensors.csv", index=False)
print("network:", len(seg), "links,", len(sensors), "pressure sensors")

# ---------------- evaluation only: does the guidance point near the real leak?
lk = [c for c in leaks.columns if c != "Timestamp"]
L = leaks.set_index("Timestamp")[lk]
persistent = [c for c in lk if (L[c] > 0).mean() > 0.5]  # small leaks present most of the year
mid = seg.set_index("id")[["xm", "ym"]]
rows = []
pred = pd.read_csv(OUTPUTS_DIR / "predictions_v2.csv", parse_dates=["Timestamp"])
eps = episodes(pred["Timestamp"], pred["target"])
eps = eps[eps["n_steps"] >= 12]
for _, e in eps.iterrows():
    w = L.loc[e.start:e.end, [c for c in lk if c not in persistent]].mean()
    leak = w.idxmax()
    window = dev.set_index("Timestamp").loc[e.start:e.end]
    r = sensor_rank_check(window, sensors[["x", "y"]], tuple(mid.loc[leak]))
    r.update(episode_start=e.start, episode_end=e.end, leaking_pipe=leak,
             mean_leak_m3h=round(float(w.max()), 2))
    rows.append(r)
check = pd.DataFrame(rows)
check.to_csv(OUTPUTS_DIR / "inspection_check_v2.csv", index=False)
print(check.to_string())

# ---------------- held-out permutation importance (descriptive; nothing is chosen from it)
model = joblib.load(MODELS_DIR / "waterguard_rf_v2.joblib")
imputer = joblib.load(MODELS_DIR / "imputer_v2.joblib")
assert list(joblib.load(MODELS_DIR / "features_v2.joblib")) == feats
out = pd.DataFrame({"feature": feats})
for name, part in [("validation", val), ("test", test)]:
    X = imputer.transform(part[feats])
    r = permutation_importance(model, X, part["target"].to_numpy(), scoring="roc_auc", n_repeats=3,
                               random_state=RANDOM_STATE, n_jobs=1)
    out[f"{name}_auc_drop_mean"] = r.importances_mean
    out[f"{name}_auc_drop_std"] = r.importances_std
    print(name, "done")
mdi = pd.read_csv(OUTPUTS_DIR / "feature_importance_v2.csv").rename(columns={"importance": "mdi_importance"})
out = out.merge(mdi, on="feature").sort_values("test_auc_drop_mean", ascending=False)
out.round(5).to_csv(OUTPUTS_DIR / "permutation_importance_v2.csv", index=False)
print(out.head(15).round(4).to_string(index=False))
print("Done.")
