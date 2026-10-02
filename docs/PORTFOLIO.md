# WaterGuard AI Saudi — Portfolio & CV kit

Everything below is supported by executed results in this repository. Do not round numbers up, and do not drop the benchmark disclaimer.

## Project title
**WaterGuard AI Saudi — ML Water-Loss Decision-Support Prototype**

## One-line description
A Saudi-motivated machine-learning prototype that flags severe water-loss periods from hydraulic SCADA data and suggests which sensors to inspect first, evaluated on the international BattLeDIM L-Town benchmark.

## One-sentence summary
Built WaterGuard AI Saudi, an ML-based water-loss decision-support research prototype using hydraulic SCADA time series, 60 engineered features, Random Forest classification, explainable alerts and sensor inspection guidance; achieved 0.950 ROC-AUC and 0.942 precision on a held-out BattLeDIM L-Town benchmark test period.

## CV — exactly two bullets
- Built a Random Forest decision-support prototype that flags severe water-loss periods from 60 engineered hydraulic SCADA features (BattLeDIM L-Town benchmark, 105,120 five-minute records); on an untouched later test period it reached 94% alert precision and ROC-AUC 0.95, with recall of 38% reported openly.
- Designed a leakage-safe chronological evaluation with validation-only threshold tuning and four baselines, then packaged the model as an application: a validated upload-and-analyse workflow (Streamlit) and a FastAPI inference API sharing one tested service layer, with per-alert explanations, sensor inspection guidance and CSV export.

## GitHub repository description (≤ 350 characters)
ML water-loss decision-support prototype: upload hydraulic SCADA data, get risk, alerts, explanations and sensor inspection guidance. Random Forest with chronological evaluation and validation-tuned threshold; Streamlit app + FastAPI API. Evaluated on the BattLeDIM L-Town benchmark — not Saudi utility data.

Suggested topics: `machine-learning` `water` `leak-detection` `scada` `time-series` `random-forest` `streamlit` `scikit-learn` `sustainability`

## Portfolio description (website card / page)
**Problem.** Water networks lose water through leaks, and utilities collect more sensor data than people can review. Saudi Arabia's National Water Strategy (MEWA) identifies reducing network losses as an improvement opportunity.

**Solution.** Upload L-Town-compatible SCADA data (pressures, flows, demands, tank level every 5 minutes). WaterGuard validates it, builds 60 hydraulic features, runs the saved model and returns risk, alerts, an explanation for each alert, a ranking of pressure sensors to inspect first and a downloadable predictions file. The verified 2018 evaluation is available as a benchmark demo. The same engine is exposed as an HTTP API.

**How it was evaluated.** On the public BattLeDIM L-Town benchmark (a simulated network; not Saudi data). Train on January–August, tune the alert threshold on August–September, test once on September–December. A first version (V1) looked fine on ROC-AUC but caught almost nothing; the redesign (V2) removed a misleading calendar feature and added 60 hydraulic features.

**Results (test period).** Precision 94.2% · Recall 38.2% · F1 0.54 · ROC-AUC 0.95. Few false alarms, but most severe time is still missed, so it is decision support, not an autonomous detector.

**What I learned.** A good metric can hide a useless model; the target definition matters as much as the algorithm; and importance plots can mislead: the "top" sensor contributed almost nothing on new data.

**Tech stack.** Python 3.13 · pandas · NumPy · scikit-learn · Plotly · Streamlit · FastAPI · pytest · Git

**Architecture.** Offline: BattLeDIM SCADA → feature engineering → chronological split → Random Forest → validation threshold → test evaluation → saved artifacts. Online: upload → validation → same 60 features → saved model → risk, alerts, explanations, inspection guidance → export; Streamlit app and FastAPI share one service layer.

**Limitations (show them).** Simulated network; one year; two severe test episodes; recall 38%; experimental severity threshold; no true leak localisation.

**Links.** Live demo: https://waterguard-ai-produc-qien.bolt.host · API docs: https://waterguard-ai-saudi-production.up.railway.app/docs · GitHub: https://github.com/emanmusheer2008-glitch/waterguard-ai-saudi

## Screenshots (in `docs/screenshots/`)
| File | Source | Shows |
|---|---|---|
| `01-overview-web.png` | Public web frontend | Hero: benchmark overview and verified metrics |
| `02-analyze-data-streamlit.png` | Streamlit | Validation → inference → timeline → alert periods (sample file) |
| `03-risk-alerts-streamlit.png` | Streamlit | Risk monitor, alert table and per-alert explanation |
| `04-inspection-sensors-streamlit.png` | Streamlit | Inspection guidance on the L-Town network map |
| `04b-sensor-intelligence-streamlit.png` | Streamlit | Impurity vs held-out permutation importance |
| `05-model-research-web.png` | Public web frontend | Metrics, confusion matrix, curves, threshold validation |
| `06-data-guide-streamlit.png` | Streamlit | Input schema and validation rules |

## Key learning to mention in essays or interviews
- Rejected an uninformative target ("any leak", true 97.8% of the time) after looking at the data.
- V1's ROC-AUC 0.86 hid a recall of 0.2% at the default threshold.
- Kept the test period untouched: the threshold was tuned on validation only.
- Found that the top impurity-importance sensor (n215) ranked #54 of 60 on held-out data.
- Kept the Saudi motivation and the benchmark evidence separate.
