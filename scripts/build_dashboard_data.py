"""Re-apply the SAVED V2 model to rebuilt features, verify it reproduces
outputs/predictions_v2.csv, and export small files the dashboard needs.

Does NOT retrain or change V2. Outputs:
  outputs/audit_report.json            data-integrity + split facts
  outputs/dashboard_context_v2.csv.gz  per-timestamp signal values + local sensitivity (test period)
  outputs/typical_conditions_v2.csv    training-period reference statistics
  outputs/curves_v2.csv                ROC and precision-recall curve points (test)
  outputs/event_summary_v2.csv         episode-level detection analysis (test)
"""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from _common import ROOT, load_v2_dataset
from waterguard.config import MODELS_DIR, OUTPUTS_DIR, SEVERE_THRESHOLD
from waterguard.evaluation import event_level_summary, point_metrics
from waterguard.target import chronological_split

N_EXPLAIN = 12  # top signals by global (impurity) importance
N_PERM = 8      # + top signals by validation permutation importance

df, feats, leaks, report = load_v2_dataset()
train, val, test = chronological_split(df)

model = joblib.load(MODELS_DIR / "waterguard_rf_v2.joblib")
imputer = joblib.load(MODELS_DIR / "imputer_v2.joblib")
saved_feats = joblib.load(MODELS_DIR / "features_v2.joblib")
threshold = float(joblib.load(MODELS_DIR / "threshold_v2.joblib"))
assert list(saved_feats) == feats, "Rebuilt feature list differs from saved features_v2"

Xt = imputer.transform(test[feats])
prob = model.predict_proba(Xt)[:, 1]
saved = pd.read_csv(OUTPUTS_DIR / "predictions_v2.csv", parse_dates=["Timestamp"])
max_diff = float(np.abs(saved["risk_probability"].to_numpy() - prob).max())
same_ts = bool((saved["Timestamp"].to_numpy() == test["Timestamp"].to_numpy()).all())
same_target = bool((saved["target"].to_numpy() == test["target"].to_numpy()).all())
print(f"Reproduction: max |prob diff| = {max_diff:.2e}, timestamps equal={same_ts}, target equal={same_target}")

m = point_metrics(saved["target"], saved["prediction"], saved["risk_probability"])
print("Verified test metrics:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})

# ---------------- audit report
lk_cols = [c for c in leaks.columns if c != "Timestamp"]
total = leaks[lk_cols].sum(axis=1)
audit = {
    "scada_validation": report,
    "leakage_rows": len(leaks),
    "leakage_missing_values": int(leaks.isna().sum().sum()),
    "leakage_duplicate_timestamps": int(leaks["Timestamp"].duplicated().sum()),
    "leakage_timestamps_match_scada": bool((leaks["Timestamp"].to_numpy() == df["Timestamp"].to_numpy()).all()),
    "share_any_positive_leak": float((leaks[lk_cols] > 0).any(axis=1).mean()),
    "share_total_leak_ge_threshold_full_year": float((total >= SEVERE_THRESHOLD).mean()),
    "percentile_of_threshold_full_year": float((total < SEVERE_THRESHOLD).mean()),
    "train_only_80th_percentile_total_leak": float(train["total_leak"].quantile(0.80)),
    "full_year_80th_percentile_total_leak": float(total.quantile(0.80)),
    "splits": {name: {"start": str(s["Timestamp"].min()), "end": str(s["Timestamp"].max()),
                      "rows": len(s), "severe_rate": float(s["target"].mean())}
               for name, s in [("train", train), ("validation", val), ("test", test)]},
    "alert_threshold_stored": threshold,
    "reproduction_max_abs_prob_diff": max_diff,
    "reproduction_timestamps_equal": same_ts,
    "reproduction_target_equal": same_target,
    "verified_test_metrics": m,
}
(OUTPUTS_DIR / "audit_report.json").write_text(json.dumps(audit, indent=2, default=str))

# ---------------- typical conditions (training, non-severe only)
imp = pd.read_csv(OUTPUTS_DIR / "feature_importance_v2.csv")
top = imp["feature"].head(N_EXPLAIN).tolist()
# Also explain the signals that matter most on HELD-OUT data (validation permutation importance,
# from scripts/build_inspection_data.py). Validation, not test, so the test period stays out of every choice.
perm_file = OUTPUTS_DIR / "permutation_importance_v2.csv"
if perm_file.exists():
    perm = pd.read_csv(perm_file).sort_values("validation_auc_drop_mean", ascending=False)
    top = list(dict.fromkeys(top + perm["feature"].head(N_PERM).tolist()))
context_cols = list(dict.fromkeys(top + ["pressure_mean", "flow_total", "demand_total", "tank_level"]))
normal = train[train["target"] == 0]
typical = pd.DataFrame({
    "feature": context_cols,
    "train_normal_median": [normal[c].median() for c in context_cols],
    "train_normal_q05": [normal[c].quantile(0.05) for c in context_cols],
    "train_normal_q95": [normal[c].quantile(0.95) for c in context_cols],
    "train_normal_std": [normal[c].std() for c in context_cols],
})
typical.to_csv(OUTPUTS_DIR / "typical_conditions_v2.csv", index=False)

# ---------------- local sensitivity: replace one signal with its typical value
# (single-feature counterfactual, all others held fixed). Descriptive only.
feat_index = {f: i for i, f in enumerate(feats)}
med = dict(zip(typical["feature"], typical["train_normal_median"]))
ctx = test[["Timestamp"] + context_cols].copy()
for f in top:
    Xmod = Xt.copy()
    Xmod[:, feat_index[f]] = med[f]
    p_mod = model.predict_proba(Xmod)[:, 1]
    ctx[f"sens_{f}"] = prob - p_mod  # >0: this signal's current value pushes risk up
num = ctx.columns.drop("Timestamp")
ctx[num] = ctx[num].round(5)
ctx.to_csv(OUTPUTS_DIR / "dashboard_context_v2.csv.gz", index=False, compression="gzip")

# ---------------- curves
fpr, tpr, roc_t = roc_curve(saved["target"], saved["risk_probability"])
prec, rec, pr_t = precision_recall_curve(saved["target"], saved["risk_probability"])
step = max(1, len(fpr) // 600)
roc = pd.DataFrame({"curve": "roc", "x": fpr[::step], "y": tpr[::step]})
step = max(1, len(prec) // 600)
pr = pd.DataFrame({"curve": "pr", "x": rec[::step], "y": prec[::step]})
pd.concat([roc, pr]).round(5).to_csv(OUTPUTS_DIR / "curves_v2.csv", index=False)

# ---------------- events
ev = event_level_summary(saved.assign(total_leak=saved["total_leak"]), min_steps=12)
ev.to_csv(OUTPUTS_DIR / "event_summary_v2.csv", index=False)
print(ev.to_string())
print("Done.")
