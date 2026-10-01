# WaterGuard AI Saudi — Portfolio & CV kit

Everything below is supported by executed results in this repository. Do not round numbers up, and do not drop the benchmark disclaimer.

## Project title
**WaterGuard AI Saudi — ML Water-Loss Decision-Support Prototype**

## One-line description
A Saudi-motivated machine-learning prototype that flags severe water-loss periods from hydraulic SCADA data and suggests which sensors to inspect first, evaluated on the international BattLeDIM L-Town benchmark.

## CV — exactly two bullets
- Built a Random Forest decision-support prototype that flags severe water-loss periods from 60 engineered hydraulic SCADA features (BattLeDIM L-Town benchmark, 105,120 five-minute records); on an untouched later test period it reached 94% alert precision and ROC-AUC 0.95, with recall of 38% reported openly.
- Designed a leakage-safe chronological train/validation/test evaluation with validation-only threshold tuning, compared it against four baselines, and shipped an 8-page Streamlit dashboard with per-alert explanations and sensor inspection guidance, backed by automated tests and fully reproducible pipelines.

## GitHub repository description (≤ 350 characters)
ML decision-support prototype that flags severe water-loss periods from hydraulic SCADA data (pressures, flows, tank level). Random Forest, chronological evaluation, validation-tuned threshold, Streamlit dashboard. Evaluated on the BattLeDIM L-Town benchmark — not Saudi utility data.

Suggested topics: `machine-learning` `water` `leak-detection` `scada` `time-series` `random-forest` `streamlit` `scikit-learn` `sustainability`

## Portfolio description (website card / page)
**Problem.** Water networks lose water through leaks, and utilities collect more sensor data than people can review. Saudi Arabia's National Water Strategy (MEWA) identifies reducing network losses as an improvement opportunity.

**Solution.** WaterGuard reads pressure, flow, demand and tank-level signals every 5 minutes and estimates whether the network is in a severe water-loss period. A dashboard explains each alert and ranks pressure sensors by how far they have dropped below normal, as a starting point for inspection.

**How it was evaluated.** On the public BattLeDIM L-Town benchmark (a simulated network; not Saudi data). Train on January–August, tune the alert threshold on August–September, test once on September–December. A first version (V1) looked fine on ROC-AUC but caught almost nothing; the redesign (V2) removed a misleading calendar feature and added 60 hydraulic features.

**Results (test period).** Precision 94.2% · Recall 38.2% · F1 0.54 · ROC-AUC 0.95. Few false alarms, but most severe time is still missed, so it is decision support, not an autonomous detector.

**What I learned.** A good metric can hide a useless model; the target definition matters as much as the algorithm; and importance plots can mislead: the "top" sensor contributed almost nothing on new data.

**Tech stack.** Python 3.13 · pandas · NumPy · scikit-learn · Plotly · Streamlit · pytest · Git

**Architecture.** BattLeDIM SCADA → validation → feature engineering → chronological split → Random Forest → validation threshold → test evaluation → saved outputs → Streamlit dashboard.

**Limitations (show them).** Simulated network; one year; two severe test episodes; recall 38%; experimental severity threshold; no true leak localisation.

**Links.** GitHub: _add after publishing_ · Live demo: _add after deploying_

## Recommended screenshots (in `docs/screenshots/`)
1. `overview.png` — headline KPIs, precision-vs-recall note, risk timeline with severe bands
2. `inspect.png` — network map with sensor deviations (the most distinctive visual)
3. `performance.png` — confusion matrix, curves, baselines
4. `sensors.png` — impurity vs held-out importance (shows critical thinking)
5. `explain.png` — per-alert explanation

## Key learning to mention in essays or interviews
- Rejected an uninformative target ("any leak", true 97.8% of the time) after looking at the data.
- V1's ROC-AUC 0.86 hid a recall of 0.2% at the default threshold.
- Kept the test period untouched: the threshold was tuned on validation only.
- Found that the top impurity-importance sensor (n215) ranked #54 of 60 on held-out data.
- Kept the Saudi motivation and the benchmark evidence separate.
