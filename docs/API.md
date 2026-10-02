# WaterGuard inference API

One FastAPI application (`api/main.py`) exposes the same inference service that the Streamlit app uses (`src/waterguard/service.py`). No model logic is duplicated: the API only parses HTTP requests and serialises results.

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --port 8000          # from the project root
# interactive docs: http://localhost:8000/docs   ·   OpenAPI schema: http://localhost:8000/openapi.json
```

Environment variable `WATERGUARD_CORS_ORIGINS` (comma-separated origins; default `*` = any origin, acceptable because the API uses no cookies or credentials) controls which browser origins may call the API. Set it to your frontend URL(s) once they are known.

**Railway:** `railway.json` sets the start command `/bin/sh -c "exec python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"` and the health check `/health`; dependencies come from `requirements.txt`.

## Architecture

```
React / Next.js frontend ──HTTP──► api/main.py (FastAPI) ──► waterguard.service ──► saved V2 artifacts (models/)
Streamlit app (app.py, ui/) ─────────────────────────────────► waterguard.service      pre-computed benchmark (outputs/)
```

The API is **stateless**: nothing is stored between requests. `/explain` therefore takes the file again together with a timestamp. Uploads are read into memory (max 100 MB), never written to disk and never executed.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness and whether the model is loaded |
| GET | `/model-info` | Model configuration, threshold, verified metrics, approved wording |
| GET | `/sample-schema` | Expected input format, column list, limits |
| GET | `/sample-files/{name}` | `template`, `sample` or `sample-ground-truth` CSV |
| POST | `/analyze` | Validate + run the saved model on an uploaded file |
| POST | `/analyze-groups` | Same as `/analyze` for four files (pressures, demands, flows, levels) |
| POST | `/explain` | Explanation + inspection guidance for one timestamp of an uploaded file |
| GET | `/benchmark` | Verified benchmark results (metrics, episodes, curves, baselines, importances) |
| GET | `/benchmark/timeline` | Benchmark risk timeline (`resolution=1h` or `5min`) with ground truth (evaluation only) |
| GET | `/benchmark/explain` | Pre-computed explanation for a benchmark timestamp |
| GET | `/network` | L-Town map geometry (pipes, pressure-sensor coordinates) |

### GET /health
```json
{"status": "ok", "model_loaded": true, "version": "3.0.0"}
```
`status` is `"degraded"` if `models/waterguard_rf_v2.joblib` is missing.

### GET /model-info
```json
{
  "name": "WaterGuard V2",
  "algorithm": "RandomForestClassifier(n_estimators=300, max_depth=16, min_samples_leaf=4, class_weight='balanced_subsample', random_state=42)",
  "n_features": 60, "features": ["P_n1", "P_n4", "…"],
  "alert_threshold": 0.22000000000000003, "alert_threshold_display": 0.22,
  "threshold_selection": "Best F1 on the validation period (8 Aug - 13 Sep 2018) only.",
  "target": "Severe water-loss period: total benchmark leakage >= 40 m3/h (experimental threshold).",
  "training_period": {"start": "2018-01-01 00:00:00", "end": "2018-08-07 23:55:00", "rows": 63072},
  "test_period": {"start": "2018-09-13 12:00:00", "end": "2018-12-31 23:55:00", "rows": 31536},
  "verified_test_metrics": {"tn": 22332, "fp": 210, "fn": 5561, "tp": 3433, "precision": 0.9424, "recall": 0.3817, "f1": 0.5433, "roc_auc": 0.9501},
  "v1_baseline_metrics": {"precision": 0.4255, "recall": 0.0022, "f1": 0.0044, "roc_auc": 0.8602},
  "dataset": "BattLeDIM 2018, L-Town benchmark network (…)",
  "wording": {"benchmark_disclaimer": "…", "precision": "…", "recall": "…", "different_network": "…"}
}
```

### GET /sample-schema
```json
{
  "format": "One table, one row per 5-minute timestamp. CSV … or XLSX …",
  "timestamp_column": "Timestamp", "sampling_interval_minutes": 5,
  "column_groups": {
    "pressures": {"prefix": "P_", "unit": "m (pressure head)", "count": 33, "columns": ["P_n1", "…"]},
    "demands":   {"prefix": "D_", "unit": "L/h", "count": 82, "columns": ["D_n1", "…"]},
    "flows":     {"prefix": "F_", "unit": "m³/h", "count": 3,  "columns": ["F_p227", "F_p235", "F_PUMP_1"]},
    "levels":    {"prefix": "L_", "unit": "m", "count": 1,  "columns": ["L_T1"]}
  },
  "total_sensor_columns": 119,
  "ground_truth": "Not required and never used for prediction. …",
  "limits": {"max_file_mb": 100, "max_rows": 110000, "min_rows": 12, "allowed_extensions": [".csv", ".xlsx"]},
  "network": "Columns must be the L-Town benchmark sensors. …",
  "sample_files": {"template": "/sample-files/template", "sample": "/sample-files/sample", "sample-ground-truth": "/sample-files/sample-ground-truth"}
}
```

### POST /analyze
`multipart/form-data`:

| Field | Required | Description |
|---|---|---|
| `file` | yes | `.csv` or `.xlsx` in the WaterGuard schema (wide table, or BattLeDIM 4-sheet workbook) |
| `ground_truth` | no | Labels file (`Timestamp` + `total_leak`, or one column per leak, m³/h). **Evaluation only.** |

Query: `include_predictions` (bool, default `true`), `top_alerts` (0–20, default 5).

```bash
curl -F "file=@samples/waterguard_sample_ltown2018_shifted_to_2026.csv" http://localhost:8000/analyze
```

**200** response (abridged, real values for the sample file):
```json
{
  "mode": "user_analysis",
  "validation": {
    "status": "ok", "issues": [], "n_rows": 1152,
    "start": "2026-10-04 00:00:00", "end": "2026-10-07 23:55:00", "interval_minutes": 5.0,
    "n_gaps": 0, "missing_steps_in_gaps": 0, "missing_cells": 0, "missing_share": 0.0,
    "missing_by_group": {"pressures": 0.0, "demands": 0.0, "flows": 0.0, "levels": 0.0},
    "columns_with_missing": {}, "ignored_columns": [], "ground_truth_like_columns": [],
    "compatibility": {"schema": "L-Town V2 (119 sensor columns)", "level": "compatible",
                      "message": "Schema matches L-Town and values lie within the range seen in training.",
                      "share_outside_training_range": 0.0, "share_outside_by_group": {"pressures": 0.0, "…": 0.0}}
  },
  "summary": {
    "n_observations": 1152, "start": "2026-10-04 00:00:00", "end": "2026-10-07 23:55:00",
    "n_alerts": 258, "alert_share": 0.2240, "max_risk": 0.7526, "max_risk_timestamp": "2026-10-06 18:40:00",
    "mean_risk": 0.1082,
    "risk_histogram": {"bin_edges": [0.0, 0.1, "…", 1.0], "counts": [740, 130, 139, 102, 32, 7, 1, 1, 0, 0]},
    "alert_periods": [{"start": "2026-10-04 10:25:00", "end": "2026-10-04 10:25:00", "steps": 1, "peak_risk": 0.2492}],
    "n_reduced_context": 6, "n_imputed_values": 0
  },
  "predictions": [
    {"timestamp": "2026-10-04 00:00:00", "risk_probability": 0.002322, "alert": false, "reduced_context": true, "imputed_values": false}
  ],
  "top_alerts": [
    {"timestamp": "2026-10-06 18:40:00", "risk_probability": 0.7526, "alert": true, "reduced_context": false, "imputed_values": false,
     "signals": [{"signal": "P_n114", "category": "Individual pressure sensor", "current_value": 52.99, "typical_median": 53.71,
                  "typical_q05": 53.12, "typical_q95": 54.39, "deviation_sigma": -1.86, "outside_typical_range": true,
                  "risk_change_if_typical": 0.1171, "global_importance_rank": 5}],
     "inspection": {"ranking": [{"rank": 1, "sensor": "n644", "deviation_sigma": -4.112}],
                    "all_deviations": {"n644": -4.112, "…": 0.0}, "sensors_below_minus_2_sigma": 26, "n_sensors": 33,
                    "note": "Inspection guidance, not leak localisation: …"}}
  ],
  "evaluation": null,
  "notices": {"not_live": "…", "different_network": "…", "inspection_guidance": "…"}
}
```
With `ground_truth`, `evaluation` becomes `{"tn", "fp", "fn", "tp", "precision", "recall", "f1", "roc_auc", "pr_auc", "matched_rows", "severe_rate", "severe_threshold_m3h": 40.0, "note": "Evaluation only: …"}`.

### POST /analyze-groups
`multipart/form-data` fields `pressures`, `demands`, `flows`, `levels` (each: `Timestamp` + unprefixed sensor IDs, CSV with `,` or `;`/decimal commas, or XLSX — e.g. BattLeDIM `2019_SCADA_Pressures.csv`), optional `ground_truth`; same query parameters and response as `/analyze`. Files are joined on `Timestamp`.

### POST /explain
`multipart/form-data`: `file` (same as `/analyze`), `timestamp` (e.g. `2026-10-06 12:00`). Returns one object shaped like an element of `top_alerts`. **404** `TIMESTAMP_NOT_FOUND` if the timestamp is not in the file.

### GET /benchmark
`{mode: "benchmark_demo", metrics, v1, splits, episodes, threshold_validation, curves: {roc: {x, y}, pr: {x, y}}, baselines, severity_sensitivity, feature_importance, inspection_check, wording}` — all values from the verified `outputs/` files.

### GET /benchmark/timeline?resolution=1h|5min
Column-oriented arrays: `timestamp, risk_probability, alert, ground_truth_total_leak_m3h, ground_truth_severe` (+ `ground_truth_note`). `1h` = 2,628 points (hourly max risk, any alert, mean leakage); `5min` = 31,536 points.

### GET /benchmark/explain?timestamp=2018-10-09 14:30
Same shape as `/explain` plus `ground_truth: {total_leak_m3h, severe, note}`. 404 outside 13 Sep – 31 Dec 2018.

### GET /network
`{links: {id[], kind[], x0[], y0[], x1[], y1[]}, pressure_sensors: [{id, x, y}], source}` — 909 links, 33 sensors.

## Errors
Every error has the same shape:
```json
{"error": {"code": "UNSUPPORTED_FORMAT", "message": "Unsupported file type. Upload a .csv or .xlsx file."}}
```

| HTTP | Code | When |
|---|---|---|
| 400 | `UNSUPPORTED_FORMAT`, `EMPTY_FILE`, `FILE_TOO_LARGE`, `MALFORMED_CSV`, `MALFORMED_XLSX`, `SHEETS_NOT_ALIGNED`, `GT_*` | The file itself cannot be read |
| 422 | `VALIDATION_FAILED` (+ `validation` report with issue codes below) | Readable but not analysable |
| 404 | `TIMESTAMP_NOT_FOUND`, unknown sample file | — |
| 422 | FastAPI request validation (e.g. bad `resolution`) | — |
| 500 | `INTERNAL_ERROR` | Unexpected; no paths or tracebacks are returned |

Validation issue codes (`level` = error / warning / info): `MISSING_TIMESTAMP`, `BAD_TIMESTAMP`, `TIMEZONE_REMOVED`, `MISSING_COLUMNS`, `GROUND_TRUTH_COLUMNS_IGNORED`, `EXTRA_COLUMNS`, `TOO_FEW_ROWS`, `TOO_MANY_ROWS`, `DUPLICATE_TIMESTAMPS`, `UNSORTED_TIMESTAMPS`, `INVALID_SAMPLING`, `IRREGULAR_SAMPLING`, `GAPS`, `NON_NUMERIC_VALUES`, `EMPTY_COLUMNS`, `MISSING_VALUES`, `VALUES_OUTSIDE_TRAINING_RANGE`.

## Safety
- Extension whitelist (`.csv`, `.xlsx`); 100 MB hard limit enforced while streaming the upload; 110,000-row limit.
- XLSX must be a real zip workbook; parsed with openpyxl + `defusedxml` (XML-bomb protection). No macros or formulas are evaluated.
- Files are parsed from memory; nothing is persisted or executed. Sample downloads use a fixed whitelist, never a user path.
- Error responses never include file-system paths or stack traces.
- Large XLSX files are slow to parse (the 92 MB BattLeDIM 2019 workbook takes about 1.5 minutes); CSV is much faster.
