# WaterGuard AI Saudi: Detecting Severe Water-Loss Periods from Hydraulic SCADA Data on the BattLeDIM L-Town Benchmark

**Author:** Eman Musheer · **Type:** independent educational research prototype · **Version:** October 2026 (v3.0 — inference application)

---

## Abstract
Water distribution networks lose water through leaks, and utilities record far more sensor data than people can review. This project asks whether a supervised machine-learning model can identify *severe water-loss periods* from hydraulic SCADA signals (pressures, flows, demands, tank level) and how reliably it does so on a later, unseen period. Using the 2018 data of the BattLeDIM L-Town benchmark (105,120 five-minute records), severity was defined experimentally as total leakage ≥ 40 m³/h. A Random Forest was trained on 60 engineered features, with a chronological train/validation/test split and an alert threshold selected on validation data only. On the untouched test period (13 Sep – 31 Dec 2018) the model reached precision 0.942, recall 0.382, F1 0.543 and ROC-AUC 0.950. It outperformed logistic-regression, isolation-forest and rule-based baselines. The test period contains only two severe episodes, recall is limited, and leak signatures shift over time, so the results show feasibility on a benchmark, not operational readiness. The project is motivated by Saudi Arabia's water-loss challenge but uses no Saudi data.

## 1. Problem
Physical losses from distribution networks waste treated water and the energy used to produce it. Leaks range from small, long-running background losses to large events that build over days. SCADA systems record pressure, flow and storage at high frequency, but turning those signals into timely, trustworthy alerts is difficult.

## 2. Saudi context (motivation, not data)
The Ministry of Environment, Water and Agriculture's National Water Strategy lists reducing losses in the network as an improvement opportunity for urban water use, estimating those losses at more than 25% in different regions (MEWA, last edited 6 July 2025). One of the strategy's objectives is to "enhance water demand management across all uses". These statements motivate the application. They are **not** evidence about this model: all experiments use an international simulated benchmark, and no claim is made about performance on Saudi networks. No current, authoritative, clearly defined national non-revenue-water figure was used, because none could be verified from a primary source during this work.

## 3. Dataset
BattLeDIM 2020 (Vrachimis et al., 2020; CC BY 4.0) provides a simulated network, L-Town, with realistic demands, sensor noise and leaks. The 2018 "historical" files were used:

| File | Content |
|---|---|
| `2018_SCADA.xlsx` | 105,120 timestamps (5 min, all of 2018): 33 pressures (m), 82 demands (L/h), 3 flows (m³/h), 1 tank level (m) |
| `2018_Leakages.csv` | Leak flow (m³/h) at 14 pipes; `;`-separated with decimal commas |
| `L-TOWN.inp` | EPANET network model (used only for the map and the inspection check) |

Validation found no duplicates, gaps or missing values, and identical timestamps across all sheets and the leakage file.

## 4. Target definition
At least one leak is non-zero in 97.8% of timestamps, because several small leaks persist for months. A binary "any leak" target was therefore rejected. Total leakage has median 23.9, 80th percentile 39.5 and maximum 80.0 m³/h. A 5-minute step is labelled **severe** if total leakage ≥ 40 m³/h (19.98% of 2018). This is an **experimental modelling threshold**, not an engineering, regulatory, competition or Saudi standard. The year contains five severe episodes longer than one hour.

## 5. Feature engineering
V2 uses 60 features computed only from current and past values: the 33 individual pressures; pressure mean, min, max, standard deviation and range; demand total, mean and standard deviation; the three flows and their total, mean and standard deviation; tank level; hour of day as sine/cosine; and 5- and 30-minute differences of pressure mean, min and standard deviation, total flow and tank level. Calendar month was excluded (see §7).

## 6. Experimental design
- **Chronological split** (no shuffling): train 60% (1 Jan – 7 Aug, 63,072 rows, 16.9% severe), validation 10% (8 Aug – 13 Sep, 10,512 rows, 13.0%), test 30% (13 Sep – 31 Dec, 31,536 rows, 28.5%).
- **Preprocessing:** a median imputer fitted on training rows only (the data has no missing values except the first rows of difference features).
- **Model:** `RandomForestClassifier(n_estimators=300, max_depth=16, min_samples_leaf=4, class_weight="balanced_subsample", random_state=42)`.
- **Threshold selection:** best F1 over 0.05–0.80 on validation → 0.22.
- **Evaluation:** a single scoring of the test period. Metrics are per 5-minute step; an episode-level analysis is added because steps are strongly autocorrelated.
- **Leakage controls:** ground truth excluded from features; causal features; train-only preprocessing; validation-only tuning. Each control is enforced by an automated test.

## 7. V1 baseline
V1 used 20 aggregate features including `hour`, `dayofweek` and `month`, a 70/30 chronological split, and the default 0.5 threshold. Test: precision 0.4255, recall 0.0022, F1 0.0044, ROC-AUC 0.8602 (TN 22,515, FP 27, FN 8,974, TP 20). The model ranked risk reasonably but almost never crossed 0.5. Its most important feature was `month` (0.229), suggesting it partly encoded the timing of 2018's leaks. V2 was designed in response.

## 8. Results (V2, test period)
| | Pred. not severe | Pred. severe |
|---|---:|---:|
| Not severe | 22,332 | 210 |
| Severe | 5,561 | 3,433 |

Precision 0.9424 · Recall 0.3817 · F1 0.5433 · ROC-AUC 0.9501 · PR-AUC 0.8806 · Accuracy 0.8170. At the selected threshold, validation gave precision 0.3109, recall 0.7036 and F1 0.4312.

**Episodes.** The test period contains two severe episodes: pipe p158 (6–23 Oct) and pipe p369 (26 Oct–8 Nov). Both were alerted at their first 5-minute step; the model stayed above threshold for 47.6% and 26.2% of their duration.

**Baselines** (same split, validation-chosen thresholds):

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Random Forest V2 | 0.9424 | 0.3817 | 0.5433 | 0.9501 |
| Always alert | 0.2852 | 1.0000 | 0.4438 | 0.5000 |
| Single signal (`pressure_std`) | 0.2853 | 1.0000 | 0.4439 | 0.6055 |
| Logistic Regression | 0.2135 | 0.0067 | 0.0129 | 0.2200 |
| Isolation Forest | 0.4567 | 0.2860 | 0.3517 | 0.7406 |

**Severity sensitivity** (retrained with the same recipe): F1 0.786 at ≥ 30, 0.678 at ≥ 35, 0.581 at ≥ 37.56 (training-only 80th percentile), 0.543 at ≥ 40, 0.469 at ≥ 45; ≥ 50 has no validation or test positives.

**Reproducibility.** Retraining from raw data reproduced the saved test probabilities (maximum absolute difference below 3e-16) and identical confusion matrices; experiment outputs were byte-identical on re-run.

## 9. Interpretation
- V2's alerts are reliable (precision 0.94), and its ranking is strong (AUC 0.95), but it misses most severe time (recall 0.38). It tends to alert at an episode's onset and then lose confidence as conditions persist.
- The gap between validation and test precision is largely explained by prevalence (13.0% vs 28.5%).
- Logistic Regression's below-random AUC indicates that the relationship between signals and severity changes between leak episodes; the forest is more robust but not immune.
- **Feature importance.** Impurity importance ranks pressure sensor n215 first (0.133). Held-out permutation importance ranks it 54th of 60 on test (AUC drop 0.002), while n229 is first on both validation (0.056) and test (0.167). n215 is nearly constant (≈ 39.09 m, 0.01 m resolution) except during the March training leak (mean 38.96 m). Impurity importance therefore overstates it. Neither measure implies causation.
- **Inspection guidance.** A model-independent heuristic ranks pressure sensors by time-of-day-adjusted drop relative to normal training periods (spread floored at 0.05 m). Averaged over each test episode, the sensor nearest the true leaking pipe ranked 1st (p158 → n644) and 2nd (p369 → n429) of 33. With two episodes and map (not hydraulic) distance, this is preliminary support for using pressure deviations to prioritise inspection. It is not a localisation result.

## 10. Dashboard
The Streamlit application has six sections: Overview; Analyze Data (upload, validation, inference, explanation, export); Risk & Alerts (monitor, detection timeline, per-alert explanation); Inspection & Sensors (network-map guidance, impurity vs permutation importance, signal explorer); Model & Research (performance, methodology, context, limitations); and Data Guide. A data-source switch separates the *benchmark demo* (retrospective replay of the test period, with ground truth labelled "evaluation only") from *your analysis* (uploaded data, no ground truth unless supplied separately). The app never retrains.

## 11. Limitations
1. Simulated network, one year, two severe test episodes (five in the year).
2. Limited recall.
3. The experimental target threshold was derived from the full-year distribution.
4. Non-stationary leak signatures.
5. No localisation; the inspection heuristic was checked on two episodes only.
6. Uncalibrated probabilities.
7. Autocorrelated step-level metrics overstate the effective sample size.

## 12. Responsible interpretation
WaterGuard AI Saudi is an independent educational research prototype and is not affiliated with or endorsed by a Saudi government entity or water utility. It has not been validated on real Saudi data and must not inform operational decisions. It replays benchmark data and is not live monitoring. Explanations describe model and sensor behaviour, not physical causes. Ground truth is used only for labels and evaluation.

## 13. Future work
Evaluation on BattLeDIM 2019; persistence/hysteresis alerting with event-level metrics; walk-forward validation; calibration on validation; model-based localisation using the hydraulic model; and, under a data-sharing agreement, evaluation on real utility data. A future operational architecture might be: SCADA stream → real-time validation and features → model → risk score → operator dashboard → field inspection → feedback labels. That architecture does not exist in this project.

## 13b. From evaluation to application
The trained model was packaged as an inference application **without retraining**. A single service layer validates uploaded SCADA data, rebuilds the 60 features with the training code, applies the saved imputer, Random Forest and 0.22 threshold, and returns risk probabilities, alerts, local explanations and inspection guidance. A Streamlit interface and a FastAPI API both use this layer.

- **Equivalence.** On the 2018 data the service reproduces the benchmark test probabilities to within 2.2e-16 with identical alerts, so the application and the evaluated model are the same function.
- **Calendar independence.** No calendar feature is used. L-Town data with timestamps shifted to 2026 yields identical probabilities after a 30-minute warm-up (the change features need history; those rows are flagged).
- **Domain shift.** Compatibility is defined by the network, not the date. Inputs must match the L-Town sensor schema, and the validator reports the share of readings outside the training range. Data from a physically different network would require network-specific labelled history, retraining, chronological validation and threshold selection.
- **Supplementary unseen-year check.** The official BattLeDIM 2019 workbook passed validation (1.4% of readings outside the training range) and was analysed end to end. Against the 2019 labels: precision 0.998, recall 0.678, F1 0.807, ROC-AUC 0.972. Total leakage in 2019 is much higher (median 75.9 vs 23.9 m³/h), so 82.1% of steps exceed the 2018-derived 40 m³/h threshold. At this prevalence, precision is uninformative ("always alert" scores 0.82). The figures show the pipeline works on a real unseen year; they are not comparable with, and do not replace, the 2018 benchmark. They also show that a severity threshold defined from one period's distribution need not transfer.

## 14. Conclusion
On a public benchmark and under a leakage-safe chronological evaluation, a Random Forest on engineered hydraulic features identified severe water-loss periods with high precision and strong ranking, but limited recall. The main contributions are methodological rather than numerical: rejecting an uninformative target, exposing a weak baseline's misleading AUC, validation-only threshold selection, transparent reporting of failure modes, and a decision-support interface that separates model output from ground truth.

## References
- Ministry of Environment, Water and Agriculture (MEWA), Kingdom of Saudi Arabia. *National Water Strategy* [web page], last edited 6 July 2025. https://www.mewa.gov.sa/en/Ministry/Agencies/TheWaterAgency/Topics/Pages/Strategy.aspx — supports: network-loss reduction is identified as an improvement opportunity, with losses estimated at more than 25% in different regions; strategic objective on demand management.
- Vrachimis, S. G., Eliades, D. G., Taormina, R., Ostfeld, A., Kapelan, Z., Liu, S., Kyriakou, M. S., Pavlou, P., Qiu, M., & Polycarpou, M. M. (2020). *Dataset of BattLeDIM: Battle of the Leakage Detection and Isolation Methods* (v1). Zenodo, published 7 Sep 2020. https://doi.org/10.5281/zenodo.4017659 — CC BY 4.0. Supports: all data, units (README), network model.
- BattLeDIM competition website. https://battledim.ucy.ac.cy/ — supports: competition purpose and organisers.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32. — method.
