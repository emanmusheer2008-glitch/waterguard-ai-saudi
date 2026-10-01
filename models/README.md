# Models

Model binaries (`*.joblib`, ~25 MB each) are git-ignored. The dashboard does not need them; it reads pre-computed results from `outputs/`.

| File | Content |
|---|---|
| `waterguard_rf_v2.joblib` | Random Forest V2 (main model) |
| `imputer_v2.joblib` | median imputer fitted on training rows only |
| `features_v2.joblib` | ordered list of the 60 feature names |
| `threshold_v2.joblib` | alert threshold chosen on validation (0.22) |
| `*_v1.joblib` | V1 baseline (model, imputer, 20 features) |

Regenerate them deterministically (`random_state=42`) from the project root:

```bash
python src/train_v1.py   # V1 baseline
python src/train_v2.py   # V2 main model (reproduces saved probabilities to within 3e-16)
```
