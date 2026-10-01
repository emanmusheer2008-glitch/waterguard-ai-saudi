# Models

| File | Committed | Content |
|---|---|---|
| `waterguard_rf_v2.joblib` | yes (25 MB) | Random Forest V2 — the reported model; needed by Analyze Data and the API |
| `imputer_v2.joblib` | yes | median imputer fitted on training rows only |
| `features_v2.joblib` | yes | ordered list of the 60 feature names |
| `threshold_v2.joblib` | yes | alert threshold chosen on validation (0.22) |
| `inference_reference_v2.json` | yes | input schema, training value ranges, explanation signals and typical values (training rows only) |
| `pressure_typical_median_v2.csv`, `pressure_typical_spread_v2.csv` | yes | usual pressure per sensor and 5-minute slot of day (non-severe training rows) — inspection guidance |
| `*_v1.joblib` | no | V1 baseline (historical); regenerate with `python src/train_v1.py` |

Regenerate deterministically (`random_state=42`) from the project root:
```bash
python src/train_v2.py                        # V2 (reproduces saved probabilities to within 3e-16)
python scripts/build_inference_reference.py   # reference files for inference
```
The model file must be loaded with the pinned scikit-learn version (1.9.1, see requirements.txt).
