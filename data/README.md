# Data

Raw files are **not** committed (size). Download them from the official BattLeDIM record:

> Vrachimis, S. G., Eliades, D. G., Taormina, R., Ostfeld, A., Kapelan, Z., Liu, S., Kyriakou, M. S., Pavlou, P., Qiu, M., & Polycarpou, M. M.
> *Dataset of BattLeDIM: Battle of the Leakage Detection and Isolation Methods.* Zenodo. https://doi.org/10.5281/zenodo.4017659 — licence CC BY 4.0.

```bash
python scripts/download_data.py
```

or download manually and place here:

| File | Size | Used for |
|---|---|---|
| `2018_SCADA.xlsx` | 92 MB | model inputs (pressures, demands, flows, tank level) |
| `2018_Leakages.csv` | 6 MB | ground truth (target + evaluation only) |

The L-Town network and these measurements are from an international research benchmark, **not** from Saudi Arabia.
Do not edit these files; `data/cache/` holds optional local pickle caches and is git-ignored.
