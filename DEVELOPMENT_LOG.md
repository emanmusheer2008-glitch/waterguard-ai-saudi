# WaterGuard AI Saudi — Development Log

This log records what actually happened while building WaterGuard, including the mistakes. Each entry is backed by something you can check: the Git history (`git show 6d4097e` is the original project), the code, or a result file in `outputs/`. Where something could not be confirmed, the entry says so.

Much of the work was done with AI assistance. This file exists so that **you** can explain each decision and each error in your own words.

---

## Phase 1 — Original project (built before 30 Sep 2026; Git commit `6d4097e`)

### 1. Initial data inspection
`src/inspect_data.py` listed the SCADA workbook's sheets (`Pressures (m)`, `Demands (L_h)`, `Flows (m3_h)`, `Levels (m)`) and previewed each one. Later checks confirmed 105,120 rows per sheet at exact 5-minute steps, with 33 pressure, 82 demand, 3 flow and 1 level signal.

### 2–4. Error: the leakage CSV would not parse with defaults
- **What happened?** The original `inspect_data.py` read the leakage file with `pd.read_csv(leak_path)` (still visible in commit `6d4097e`).
- **Why?** The file was written with European conventions: `;` separates columns and `,` is the decimal mark (`1,5` means 1.5). Pandas expects `,` as the separator, so the whole header becomes one column. The first data row containing a decimal comma then appears to have two fields, and pandas stops with `ParserError: Expected 1 fields in line 2312, saw 2`. (Re-run on 1 Oct 2026. An earlier note said the read "silently produces one text column"; that was wrong and has been corrected.)
- **How was it identified?** By opening the raw file: the first line is `Timestamp;p31;p158;…`.
- **How was it fixed?** The later scripts (`analyze_leaks.py`, `severity_analysis.py`, `train_v1.py`, `train_v2.py`) all use `pd.read_csv(..., sep=";", decimal=",", parse_dates=["Timestamp"])`, with a comment explaining why. The old exploration script kept the bug until the portfolio upgrade (entry 22). No model was ever trained on wrongly parsed data.
- **What did we learn?** Always look at a raw file before trusting a loader. Two tests now guard this: one on a small sample and one on the real file, which must fail at line 2312 with defaults.

### 5–6. Discovery: "any leak > 0" is a useless target
- `analyze_leaks.py` showed that at least one of the 14 leaks is above zero in **97.8%** of timestamps. Small leaks run for months: p257 for 97.8% of the year, p427 for 87.3%, and p654 and p810 from July onward.
- A target that is "yes" 98% of the time cannot teach a model anything useful, and a model that always said "leak" would look accurate. This target was rejected.
- **Lesson:** the target definition is a modelling decision that needs as much thought as the model.

### 7–8. Severity analysis and the experimental ≥ 40 target
- `severity_analysis.py` described total leakage (sum of the 14 leaks): mean 25.52, median 23.91, 75th percentile 33.30, **80th 39.55**, 90th 46.50, 95th 57.12, max 80.04 (units m³/h, per the BattLeDIM README).
- The project labels a timestamp **severe** when total leakage ≥ 40, roughly the upper fifth (19.98% of 2018).
- **It is an experimental threshold**, not an engineering, BattLeDIM, utility, regulatory or Saudi standard. It was read from the full-year distribution, a weakness examined in entry 24.

### 9–12. V1: a model that ranked well but did not work
- `train_v1.py`: Random Forest (200 trees, depth 14, min leaf 5, `class_weight="balanced"`), 20 aggregate features including `hour`, `dayofweek` and `month`, chronological 70/30 split, default 0.5 threshold.
- **Result:** TN 22,515 · FP 27 · FN 8,974 · TP 20 → precision 0.4255, **recall 0.0022**, F1 0.0044, **ROC-AUC 0.8602**. Only 47 alerts in 3½ months.
- **Why could ROC-AUC look decent while recall was terrible?** ROC-AUC measures *ranking* over all possible thresholds. V1 usually gave severe moments higher scores than normal ones, but almost never above 0.5, so at the default threshold it stayed silent.
- **Month dominance:** `month` was the top feature (importance 0.229). Leaks don't follow the calendar, so this suggested V1 partly memorised *when* 2018's leaks happened. That would not carry over to a new year.
- **Lesson:** one good-looking metric can hide a useless system. V1 is kept in the project as an honest baseline.

### 13–19. V2: redesign
- **Features (60):** all 33 individual pressure sensors (local drops disappear in averages), pressure/demand/flow summaries, the 3 individual flows, tank level, cyclical hour (`sin`/`cos`), and 5- and 30-minute changes. **Month removed.**
- **Split:** chronological 60 / 10 / 30. Train ends 2018-08-07 23:55, validation runs to 2018-09-13 11:55, test runs to 31 Dec.
- **Model:** Random Forest, 300 trees, depth 16, min leaf 4, `class_weight="balanced_subsample"`, seed 42. The median imputer is fitted on training rows only.
- **Threshold tuning:** the best validation F1 on a 0.05–0.80 grid was at **0.22** (validation P 0.3109, R 0.7036, F1 0.4312). The stored value is `0.22000000000000003` because of how `np.arange` adds floats; this does not change any prediction.
- **Test result:** TN 22,332 · FP 210 · FN 5,561 · TP 3,433 → **precision 0.9424, recall 0.3817, F1 0.5433, ROC-AUC 0.9501**.

### 20. The trade-off
V2 alerts are usually right (94% precision), but it misses most severe 5-minute steps (38% recall). It is a decision-support aid, not an autonomous detector.

### 21–22. First dashboard and the Streamlit warnings
- The original `app.py` (437 lines) had a sidebar with 6 pages, emoji headings and an About page that already stated that the data was not Saudi.
- It called `st.plotly_chart(..., use_container_width=True)` and `st.dataframe(..., use_container_width=True)` five times. That parameter is deprecated in current Streamlit and produced warnings on every run.

---

## Phase 2 — Portfolio upgrade (30 Sep – 1 Oct 2026; commit `b33c6f4`)

### 22 (cont.). Fixes
- `use_container_width=True` replaced with `width="stretch"`. The test suite runs with `-W error::DeprecationWarning` and passes.
- The `inspect_data.py` parsing bug fixed.
- The dashboard redesigned with top navigation and 7 pages, a permanent "Retrospective replay · not live" badge, and ground truth labelled "evaluation only" everywhere. The 40 line had been labelled "V1 severe-loss definition" and was relabelled as the experimental threshold.

### 23. Data-leakage audit (the ML kind)
| Check | Result |
|---|---|
| Ground truth used as a feature? | No. `total_leak` and `target` are excluded (a test checks every feature name) |
| Features use future values? | No. Only `diff()` of earlier rows; a test changes future rows and checks that earlier features don't move |
| Imputer fitted on test? | No. Fitted on training rows only |
| Threshold tuned on test? | No. Validation only; a test checks that 0.22 is the validation-best F1 |
| Timestamp alignment | V1/V2 combine sheets by row position; verified safe because all sheets and the leakage file have identical timestamps |
| Test used more than once for choices? | No. The test-set threshold sweep is saved for reference only |

**Outcome: no leakage found, so the original V2 results stand.**

### 24. Weaknesses found and documented (none changed V2)
- **Severity threshold from the full year.** 40 ≈ the 80th percentile of all 2018 leakage, test period included. This affects the label definition, not the features. Check: the training-only 80th percentile is 37.56, and retraining with it gives P 0.9473 / R 0.4194 / F1 0.5814 / AUC 0.9530, which is similar.
- **Only two test episodes.** The 8,994 severe test steps are two leaks: p158 (6–23 Oct) and p369 (26 Oct–8 Nov). V2 alerted at the first step of both and covered 47.6% and 26.2% of them.
- **Prevalence shift.** Train 16.9%, validation 13.0%, test 28.5% severe. This explains validation precision 0.31 versus test precision 0.94.
- **Baselines.** "Always alert" scores F1 0.444, so F1 alone flatters little. Logistic Regression ranks the test period worse than random (AUC 0.22), a sign that leak signatures change over time. Isolation Forest scores AUC 0.74.
- **Reproducibility.** Retraining V2 from scratch reproduced the saved probabilities (max difference 2.2e-16).

### 25. Deployment problem (environment, not code)
`scripts/download_data.py` could not be tested in that session because the sandbox's network proxy blocked Zenodo (HTTP 403). It was tested successfully in Phase 3.

---

## Phase 3 — Final audit and completion (1 Oct 2026)

### 26. Which folder is the real project?
- **What happened?** The folder attached to this session was `%USERPROFILE%\Desktop\waterguard-ai-saudi`.
- **How identified?** It contains only empty `data/ models/ outputs/ src/ tests/` folders and a `.venv`, with no `.git`, `docs` or `scripts`. `%USERPROFILE%\OneDrive\Desktop\waterguard-ai-saudi` has the Git history (`6d4097e`, `b33c6f4`), data, models and outputs.
- **Decision:** the OneDrive folder is canonical. The Desktop skeleton was not modified.
- **Lesson:** OneDrive's Desktop redirection makes two "Desktop" folders. Check for `.git` before working.

### 27. Everything re-run from raw data
- `train_v1.py` and `train_v2.py` were re-run from the raw files in a clean copy. They gave identical confusion matrices, metrics and importances, and probabilities within 2.2e-16. The validation threshold was again 0.22.
- `run_experiments.py` was re-run, and every experiment file came out **byte-identical** to the saved one.
- `download_data.py` was run against Zenodo. The downloaded files have the **same MD5 checksums** as the local copies. It now also checks file sizes and downloads the network model `L-TOWN.inp`.
- The BattLeDIM README states leakage units as **m³/h**, so the threshold is now written as 40 m³/h.

### 28. Finding: the top feature is not what drives held-out performance
- **What happened?** `P_n215` is the #1 feature by impurity importance (0.133). Inspection showed n215 is almost constant: about 39.09 m in 0.01 m steps, with a total range of 0.2 m.
- **Why does the forest use it?** During the severe March leak (pipe p673, a *training* episode), n215 averaged 38.96 m. In every other severe episode of the year it stayed at about 39.09 m. Impurity importance is computed on training data, so it rewards a split that only helps for that one episode.
- **How identified?** Class-wise statistics per split, then **held-out permutation importance**: shuffle one feature on validation or test data and measure the ROC-AUC drop. On test, n215 ranks **#54 of 60** (drop 0.002). `P_n229` ranks #1 on both validation (0.056) and test (0.167).
- **What changed?** Nothing in the model. The permutation importances are now saved (`outputs/permutation_importance_v2.csv`) and shown next to impurity importance on the Sensor Intelligence page. The Alert Explainer now also covers the top held-out signals, chosen on **validation**, not test.
- **Lesson:** impurity importance describes what a forest used to fit its training data. Held-out permutation importance is a better guide to what drives the reported results. Neither shows causation.

### 29. New: inspection guidance ("where should operators look first?")
- The model predicts *when*, not *where*. A separate, transparent heuristic was added. Each pressure sensor is compared with its usual value at the same time of day (median of non-severe training periods), and the biggest drops are ranked. A spread floor of 0.05 m stops near-constant sensors like n215 from producing infinite scores; that problem was found when the first attempt returned `-inf`.
- **Checked against ground truth (evaluation only)** with the network model's coordinates. Over each severe test episode, the sensor nearest the true leaking pipe ranked **#1 of 33** (p158 → n644) and **#2 of 33** (p369 → n429).
- Presented as *inspection guidance, not localisation*: it covers two episodes, uses map distance, and runs on a simulated network.

### 30. Smaller fixes in this pass
- Code comments and docs corrected about how the CSV fails with defaults (entry 2).
- `dashboard_context_v2.csv` (9.4 MB after adding signals) is now saved gzip-compressed (2.8 MB), and the committed outputs total about 8 MB.
- Page renamed to "Network Overview", with plain-language explanations of what a severe-loss period is and why 94% precision is not 94% detection.
- Methodology page states the Saudi-versus-benchmark split side by side, adds the required disclaimer, and labels the real-world architecture as future work.

### 31. Deployment status
Not deployed yet; you will publish to GitHub later. Everything is prepared for Streamlit Community Cloud (see README §17).
