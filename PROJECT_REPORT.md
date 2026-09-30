# PROJECT_REPORT — WaterGuard AI Saudi upgrade (1 Oct 2026)

This is a factual log of what existed, what was checked, what changed, and what was actually executed.

## 1. What existed before this work
Location: `C:\Users\shams\OneDrive\Desktop\waterguard-ai-saudi`. A second folder, `C:\Users\shams\Desktop\waterguard-ai-saudi`, contains only empty `data/ models/ outputs/ src/ tests/` folders and a `.venv`. It was left untouched.

| Item | State |
|---|---|
| `app.py` | Working Streamlit app with 6 sidebar pages; used `use_container_width=True` (deprecated) |
| `src/` | `inspect_data.py`, `analyze_leaks.py`, `severity_analysis.py`, `train_v1.py`, `train_v2.py` |
| `data/` | `2018_SCADA.xlsx` (92.2 MB), `2018_Leakages.csv` (6.2 MB) |
| `models/` | V1/V2 Random Forests (~25 MB each), imputers, feature lists, `threshold_v2.joblib` |
| `outputs/` | predictions, importances and validation-threshold table for V1/V2 |
| `tests/` | empty |
| Git, README, requirements, .gitignore | none |
| `.venv` | Windows venv, Python 3.13.15: streamlit 1.64.0, pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, plotly 7.1.0, joblib 1.6.0, openpyxl 3.1.5 |

## 2. Files inspected
Everything listed above: all five scripts in full, `app.py` in full, all output CSVs, the joblib artifacts (loaded), and both data files (fully loaded and validated).

## 3. Audit findings

| Check | Result |
|---|---|
| Duplicate timestamps | None in any SCADA sheet or the leakage file |
| Missing values | None |
| Time gaps | None; exactly 5-minute steps, 105,120 rows |
| Sheet alignment | V1/V2 concatenate sheets by row position. **Verified safe**: all four sheets and the leakage file have identical timestamps row by row |
| Ground truth used as a feature | No. Only `total_leak`/`target` are derived from leakage, and they are excluded from features (now also a test) |
| Temporal leakage in features | No. `diff()` uses only past rows (now also a test) |
| Preprocessing fit on test | No. The imputer is fitted on training data only |
| Threshold tuning | Correct. Chosen on validation only (best F1 = 0.4312 at 0.22; stored value `0.22000000000000003` from `np.arange`, which does not affect predictions) |
| Verified V1/V2 metrics | **Reproduced exactly** from `outputs/` (V2: TN 22,332 / FP 210 / FN 5,561 / TP 3,433; P 0.9424, R 0.3817, F1 0.5433, AUC 0.9501. V1: P 0.4255, R 0.0022, F1 0.0044, AUC 0.8602) |
| Saved model vs predictions file | Re-applying the saved V2 model to rebuilt features: max probability difference 2.2e-16 |
| Determinism | Retraining V2 from scratch (same recipe, seed 42) reproduces the saved probabilities (max diff 2.2e-16) |

**Issues found (none change the verified results):**
1. **Bug in `src/inspect_data.py`**: the leakage CSV was read with pandas defaults, which produces one text column. *Impact:* exploration output only; training scripts were unaffected. *Fix:* now reads with `sep=";", decimal=","`.
2. **Deprecated Streamlit API** (`use_container_width`). *Fix:* replaced with `width="stretch"`. The test suite runs with DeprecationWarnings treated as errors and passes.
3. **Severity threshold chosen from the full-year distribution** (40 ≈ 80.0th percentile of all 2018 total leakage, test period included). *Impact:* this affects only the label definition, not the features. *Check:* the training-only 80th percentile is 37.56, and retraining with it gives P 0.9473 / R 0.4194 / F1 0.5814 / AUC 0.9530, which is similar. Documented; V2 unchanged.
4. **Metrics are per 5-minute step, and steps are strongly autocorrelated.** The 8,994 severe test steps come from only **two** episodes (6–23 Oct, 26 Oct–8 Nov). Episode analysis was added: both were alerted at their first step, with 47.6% and 26.2% of their duration covered.
5. **Prevalence shift**: train 16.9% / validation 13.0% / test 28.5% severe. This explains the gap between validation precision (0.31) and test precision (0.94). Documented.
6. The original dashboard labelled the 40 line as the "V1 severe-loss definition" and presented "Severe-Loss Periods" (really 5-minute observations). Wording corrected.

## 4. Files changed
| File | Change |
|---|---|
| `app.py` | Full redesign. Top navigation, 7 pages (new: Alert Explainer), consistent theme, ground truth always labelled "evaluation only", status badge "Retrospective replay · not live", what-if threshold clearly marked exploratory, CSV export, `width="stretch"`. Includes a `WATERGUARD_TEST_PAGE` env hook used only by tests |
| `src/inspect_data.py` | CSV parsing fix (issue 1) |

**Unchanged on purpose:** `src/train_v1.py`, `src/train_v2.py`, `analyze_leaks.py`, `severity_analysis.py`, all original `outputs/*.csv`, all `models/*.joblib`, and `data/*`.

## 5. Files created
- `src/waterguard/` (`config`, `data`, `features`, `target`, `evaluation`, `dashboard_data`): importable pipeline that replicates V1/V2 exactly
- `scripts/build_dashboard_data.py`, `scripts/run_experiments.py`, `scripts/download_data.py`, `scripts/_common.py`
- `outputs/audit_report.json`, `dashboard_context_v2.csv`, `typical_conditions_v2.csv`, `curves_v2.csv`, `event_summary_v2.csv`
- `outputs/experiments/` (`baseline_comparison.csv`, `severity_threshold_sensitivity.csv`, `v2_test_threshold_sensitivity.csv`, `experiments_summary.json`)
- `tests/` (`conftest.py`, `test_data.py`, `test_features.py`, `test_outputs_and_models.py`, `test_app.py`)
- `README.md`, `LEARNING_GUIDE.md`, `PROJECT_REPORT.md`, `data/README.md`, `models/README.md`
- `requirements.txt`, `requirements-dev.txt`, `.gitignore`, `.streamlit/config.toml`, `docs/screenshots/*.png` (7 real screenshots)

## 6. Experiments executed (all V2 thresholds chosen on validation; V2 not replaced)
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Random Forest V2 | 0.9424 | 0.3817 | 0.5433 | 0.9501 |
| Always alert | 0.2852 | 1.0000 | 0.4438 | 0.5000 |
| Single signal `pressure_std` | 0.2853 | 1.0000 | 0.4439 | 0.6055 |
| Logistic Regression | 0.2135 | 0.0067 | 0.0129 | 0.2200 |
| Isolation Forest | 0.4567 | 0.2860 | 0.3517 | 0.7406 |

Severity-definition sensitivity (retrained): at ≥30, F1 0.7864; at ≥35, 0.6775; at ≥37.56, 0.5814; at ≥40, 0.5433; at ≥45, 0.4688; at ≥50 there are no positives. The test-set threshold sweep is saved for reference only and was not used for any choice.

## 7. Tests executed
Environment: a cloud replica with **identical package versions** to the project `.venv` (Python 3.13.7 vs. 3.13.15 locally). They have not yet been run on the Windows machine itself; see "Manual actions".

| Scenario | Command | Result |
|---|---|---|
| Full project (data + models present) | `python -m pytest -q` | **33 passed, 0 failed, 0 skipped** |
| Same, with DeprecationWarnings as errors | `python -m pytest -q -W error::DeprecationWarning` | **33 passed** |
| Fresh-clone simulation (no raw data, no models) | `python -m pytest -q -rs` | **29 passed, 4 skipped** (data/model-dependent tests), 0 failed |

Dashboard validation: served headless with `streamlit run app.py`. All 7 pages were loaded in Chromium and screenshotted, with no Python warnings or tracebacks in the server log. The only browser console errors were Google Fonts being blocked by the build sandbox's network, which does not happen on a normal connection.

## 8. Deployment status
**Not deployed.** The repository is prepared for Streamlit Community Cloud: the app needs only `outputs/` (~10 MB committed), no raw data and no model binaries. Deployment needs your GitHub and Streamlit logins. `scripts/download_data.py` could not be tested from the build sandbox because Zenodo was blocked there (HTTP 403 from the sandbox proxy); the URL pattern follows Zenodo's standard `records/<id>/files/<name>` form.

## 9. Data licence decision
BattLeDIM is **CC BY 4.0** (Zenodo record 4017659), so redistribution with attribution is permitted. The raw files are nevertheless **excluded from Git** because of their size (the 92 MB workbook is close to GitHub's 100 MB hard limit and above its 50 MB warning). A download script and a citation are provided instead.

## 10. Unresolved limitations
Only two severe test episodes; recall 38%; simulated network; experimental severity threshold; no localisation; uncalibrated probabilities; the non-stationarity shown by Logistic Regression.

## 11. Recommended next improvements
1. Evaluate on the BattLeDIM 2019 files (a second, independent year).
2. Episode-aware alerting (persistence/hysteresis), measured with event-level metrics.
3. Walk-forward validation across the year for more than one test window.
4. Probability calibration (isotonic/Platt on validation).
5. Leak localisation using the L-Town `.inp` hydraulic model.
6. Add a LICENSE file of your choice for your own code (e.g. MIT).
