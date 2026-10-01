# Data

Raw files are **not** committed (size). Download them from the official BattLeDIM record:

> Vrachimis, S. G., Eliades, D. G., Taormina, R., Ostfeld, A., Kapelan, Z., Liu, S., Kyriakou, M. S., Pavlou, P., Qiu, M., & Polycarpou, M. M. (2020).
> *Dataset of BattLeDIM: Battle of the Leakage Detection and Isolation Methods* (v1). Zenodo. https://doi.org/10.5281/zenodo.4017659 — licence CC BY 4.0.

```bash
python scripts/download_data.py
```

or download manually from the Zenodo record and place here:

| File | Size | Used for |
|---|---|---|
| `2018_SCADA.xlsx` | 92 MB | model inputs: pressures (m), demands (L/h), flows (m³/h), tank level (m) |
| `2018_Leakages.csv` | 6 MB | ground truth, leak flow in m³/h (target + evaluation only). `;` separator, `,` decimals |
| `L-TOWN.inp` | 0.4 MB | EPANET network model: network map and inspection-guidance check only |

The L-Town network and these measurements come from an international research benchmark, **not** from Saudi Arabia.
Do not edit these files. `data/cache/` holds optional local pickle caches (faster re-runs) and is git-ignored.
