"""Supplementary experiments. V2 itself is NOT replaced by anything here.

1. Deterministic retrain check of V2 (same code path, same seed).
2. Baselines on the identical split, thresholds chosen on validation only:
   - "always alert" prevalence baseline
   - single-signal rule (pressure_std)
   - Logistic Regression (standardised, same 60 features)
   - Isolation Forest (unsupervised anomaly score, fit on training SCADA only)
3. V2 threshold sensitivity on the test set (reporting only - not used to choose).
4. Severity-definition sensitivity: retrain the V2 recipe with other experimental
   targets (30, 35, 45, 50 and the training-only 80th percentile).
Outputs go to outputs/experiments/.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from _common import load_v2_dataset
from waterguard.config import MODELS_DIR, OUTPUTS_DIR, RANDOM_STATE
from waterguard.evaluation import event_level_summary, point_metrics
from waterguard.target import chronological_split

OUT = OUTPUTS_DIR / "experiments"
OUT.mkdir(parents=True, exist_ok=True)
GRID = np.arange(.05, .81, .01)


def rf():
    return RandomForestClassifier(n_estimators=300, max_depth=16, min_samples_leaf=4,
                                  class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)


def pick_threshold(y_val, s_val, grid):
    best_t, best_f1 = None, -1
    for t in grid:
        f1 = point_metrics(y_val, (s_val >= t).astype(int))["f1"]
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return float(best_t), best_f1


def evaluate(name, s_val, s_test, y_val, y_test, grid, test_df):
    t, vf1 = pick_threshold(y_val, s_val, grid)
    pred = (s_test >= t).astype(int)
    m = point_metrics(y_test, pred, s_test)
    ev = event_level_summary(test_df.assign(prediction=pred, risk_probability=s_test))
    m.update(model=name, val_threshold=t, val_f1=vf1,
             episodes_detected=f"{int(ev['detected'].sum())}/{len(ev)}" if len(ev) else "0/0")
    return m


df, feats, leaks, _ = load_v2_dataset()
train, val, test = chronological_split(df)
yv, yt = val["target"].to_numpy(), test["target"].to_numpy()
imp = SimpleImputer(strategy="median").fit(train[feats])
Xtr, Xv, Xt = (imp.transform(s[feats]) for s in (train, val, test))
test_df = test[["Timestamp", "target", "total_leak"]].reset_index(drop=True)
results = []

# 1. deterministic retrain
t0 = time.time()
model = rf().fit(Xtr, train["target"])
p_re = model.predict_proba(Xt)[:, 1]
saved = pd.read_csv(OUTPUTS_DIR / "predictions_v2.csv")
retrain_diff = float(np.abs(p_re - saved["risk_probability"].to_numpy()).max())
print(f"Retrain V2: {time.time()-t0:.0f}s, max |prob diff| vs saved = {retrain_diff:.2e}")
results.append(evaluate("Random Forest V2 (retrained, same recipe)", model.predict_proba(Xv)[:, 1], p_re, yv, yt, GRID, test_df))

# 2a. always alert
m = point_metrics(yt, np.ones_like(yt), None); m.update(model="Always alert (prevalence)", val_threshold=None, val_f1=None, roc_auc=0.5, pr_auc=float(yt.mean()), episodes_detected="2/2")
results.append(m)

# 2b. single signal: pressure_std (higher std -> higher risk, direction checked on train)
col = feats.index("pressure_std")
sign = 1 if np.corrcoef(Xtr[:, col], train["target"])[0, 1] >= 0 else -1
s_v, s_t = sign * Xv[:, col], sign * Xt[:, col]
grid = np.quantile(sign * Xtr[:, col], np.linspace(0.01, 0.99, 197))
results.append(evaluate(f"Single-signal rule (pressure_std, sign {sign:+d})", s_v, s_t, yv, yt, grid, test_df))

# 2c. logistic regression
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
lr.fit(Xtr, train["target"])
results.append(evaluate("Logistic Regression (60 features)", lr.predict_proba(Xv)[:, 1], lr.predict_proba(Xt)[:, 1], yv, yt, GRID, test_df))

# 2d. isolation forest (labels NOT used for fitting; only for threshold on validation)
iso = IsolationForest(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1).fit(Xtr)
a_tr, a_v, a_t = (-iso.score_samples(X) for X in (Xtr, Xv, Xt))
grid = np.quantile(a_tr, np.linspace(0.01, 0.99, 197))
results.append(evaluate("Isolation Forest (unsupervised anomaly score)", a_v, a_t, yv, yt, grid, test_df))

res = pd.DataFrame(results)
cols = ["model", "val_threshold", "val_f1", "precision", "recall", "f1", "roc_auc", "pr_auc", "tp", "fp", "fn", "tn", "episodes_detected"]
res[cols].to_csv(OUT / "baseline_comparison.csv", index=False)
print(res[cols].round(4).to_string())

# 3. V2 threshold sensitivity on test (report only)
p = saved["risk_probability"].to_numpy(); y = saved["target"].to_numpy()
rows = []
for t in GRID:
    m = point_metrics(y, (p >= t).astype(int)); m["threshold"] = round(float(t), 2); rows.append(m)
pd.DataFrame(rows).to_csv(OUT / "v2_test_threshold_sensitivity.csv", index=False)

# 4. severity-definition sensitivity
train_p80 = float(train["total_leak"].quantile(0.80))
sev_rows = []
for thr in [30.0, 35.0, 40.0, 45.0, 50.0, round(train_p80, 2)]:
    y_tr = (train["total_leak"] >= thr).astype(int); y_v = (val["total_leak"] >= thr).astype(int).to_numpy(); y_te = (test["total_leak"] >= thr).astype(int).to_numpy()
    if y_te.sum() == 0 or y_v.sum() == 0:
        sev_rows.append({"severity_threshold": thr, "note": "no positives in validation or test"}); continue
    mdl = model if thr == 40.0 else rf().fit(Xtr, y_tr)
    r = evaluate(f"RF @ total_leak>={thr}", mdl.predict_proba(Xv)[:, 1], mdl.predict_proba(Xt)[:, 1], y_v, y_te, GRID,
                 test_df.assign(target=y_te))
    r.update(severity_threshold=thr, train_rate=float(y_tr.mean()), val_rate=float(y_v.mean()), test_rate=float(y_te.mean()))
    sev_rows.append(r); print(thr, round(r["precision"], 4), round(r["recall"], 4), round(r["f1"], 4), round(r["roc_auc"], 4))
sev = pd.DataFrame(sev_rows)
sev.to_csv(OUT / "severity_threshold_sensitivity.csv", index=False)

(OUT / "experiments_summary.json").write_text(json.dumps({
    "retrain_max_abs_prob_diff_vs_saved": retrain_diff,
    "train_only_80th_percentile_total_leak": train_p80,
    "note": "V2 remains the primary model; none of these experiments replace it.",
}, indent=2))
print("Done.")
