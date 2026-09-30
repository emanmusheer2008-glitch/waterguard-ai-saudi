# Models

Model binaries (`*.joblib`, ~25 MB each) are git-ignored. Regenerate them (deterministic, `random_state=42`) with:

```bash
python src/train_v1.py   # V1 baseline
python src/train_v2.py   # V2 main model (retraining reproduces saved probabilities to within 2.2e-16)
```

The dashboard does not need these files; it reads pre-computed results from `outputs/`.
