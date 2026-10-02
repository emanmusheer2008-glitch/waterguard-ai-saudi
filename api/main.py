"""WaterGuard inference API (FastAPI) for a future React/Next.js frontend.

All model logic lives in src/waterguard/service.py; this file only handles HTTP.
Run from the project root:
    uvicorn api.main:app --port 8000
Interactive docs: http://localhost:8000/docs
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from waterguard import __version__  # noqa: E402
from waterguard import service as svc  # noqa: E402
from waterguard.config import OUTPUTS_DIR, SEVERE_THRESHOLD, V1_VERIFIED, V2_VERIFIED  # noqa: E402
from waterguard.wording import WORDING  # noqa: E402

SAMPLES = {  # whitelist: name -> file (never a user-supplied path)
    "template": ROOT / "samples" / "waterguard_input_template.csv",
    "sample": ROOT / "samples" / "waterguard_sample_ltown2018_shifted_to_2026.csv",
    "sample-ground-truth": ROOT / "samples" / "waterguard_sample_ground_truth_shifted_to_2026.csv",
}
CHUNK = 1024 * 1024

app = FastAPI(
    title="WaterGuard AI Saudi — inference API",
    version=__version__,
    description="ML water-loss decision-support research prototype. Model trained and evaluated on the BattLeDIM "
                "L-Town benchmark (simulated network); not Saudi utility data; not live monitoring.",
)
# Allowed browser origins, comma-separated (e.g. "https://my-frontend.example"). Unset or "*" allows any
# origin; this API uses no cookies or credentials, so that is safe for a public demo. Restrict it in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("WATERGUARD_CORS_ORIGINS", "*").split(",") if o.strip()] or ["*"],
    allow_methods=["GET", "POST"], allow_headers=["*"],
)


# ------------------------------------------------------------------ helpers
def jsonable(o):
    """numpy/pandas -> plain JSON types; NaN/inf -> null."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(o))
    if o is pd.NaT or o is pd.NA:
        return None
    return o


def error(status: int, code: str, message: str, **extra) -> JSONResponse:
    return JSONResponse(status_code=status, content=jsonable({"error": {"code": code, "message": message, **extra}}))


@app.exception_handler(svc.InputError)
async def _input_error(_: Request, exc: svc.InputError):
    return error(400, exc.code, exc.message)


@app.exception_handler(Exception)
async def _unexpected(_: Request, exc: Exception):  # never leak paths or tracebacks
    return error(500, "INTERNAL_ERROR", "Unexpected server error.")


async def read_upload(f: UploadFile) -> bytes:
    """Read into memory with a hard size limit; nothing is written to disk."""
    buf = bytearray()
    while chunk := await f.read(CHUNK):
        buf.extend(chunk)
        if len(buf) > svc.MAX_UPLOAD_BYTES:
            raise svc.InputError("FILE_TOO_LARGE", f"The file is larger than {svc.MAX_UPLOAD_BYTES // 2**20} MB.")
    return bytes(buf)


def predictions_payload(res: svc.AnalysisResult) -> list[dict]:
    p = res.results
    return [{"timestamp": str(t), "risk_probability": round(float(r), 6), "alert": bool(a),
             "reduced_context": bool(rc), "imputed_values": bool(iv)}
            for t, r, a, rc, iv in zip(p["Timestamp"], p["risk_probability"], p["alert"],
                                       res.flags["reduced_context"], res.flags["imputed_values"])]


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    ok = svc.model_available()
    if ok:
        svc.load_bundle()
    return {"status": "ok" if ok else "degraded", "model_loaded": ok, "version": __version__}


@app.get("/model-info")
def model_info():
    b = svc.load_bundle()
    return jsonable({
        "name": "WaterGuard V2",
        "algorithm": "RandomForestClassifier(n_estimators=300, max_depth=16, min_samples_leaf=4, "
                     "class_weight='balanced_subsample', random_state=42)",
        "n_features": len(b.features), "features": b.features,
        "alert_threshold": b.threshold, "alert_threshold_display": 0.22,
        "threshold_selection": "Best F1 on the validation period (8 Aug - 13 Sep 2018) only.",
        "target": f"Severe water-loss period: total benchmark leakage >= {SEVERE_THRESHOLD:g} m3/h (experimental threshold).",
        "training_period": b.ref["training_period"],
        "test_period": {"start": "2018-09-13 12:00:00", "end": "2018-12-31 23:55:00", "rows": 31536},
        "verified_test_metrics": V2_VERIFIED, "v1_baseline_metrics": V1_VERIFIED,
        "dataset": "BattLeDIM 2018, L-Town benchmark network (Vrachimis et al., 2020, doi:10.5281/zenodo.4017659, CC BY 4.0)",
        "wording": WORDING,
    })


@app.get("/sample-schema")
def sample_schema():
    return jsonable({**svc.schema_description(), "sample_files": {k: f"/sample-files/{k}" for k in SAMPLES}})


@app.get("/sample-files/{name}")
def sample_file(name: str):
    if name not in SAMPLES:
        raise HTTPException(404, "Unknown sample file")
    return FileResponse(SAMPLES[name], media_type="text/csv", filename=SAMPLES[name].name)


@app.post("/analyze")
async def analyze(file: UploadFile = File(..., description="SCADA data (.csv or .xlsx) in the WaterGuard schema"),
                  ground_truth: UploadFile | None = File(None, description="Optional labels, evaluation only"),
                  include_predictions: bool = Query(True),
                  top_alerts: int = Query(5, ge=0, le=20)):
    content = await read_upload(file)
    gt = (await read_upload(ground_truth), ground_truth.filename) if ground_truth is not None and ground_truth.filename else None
    res = await run_in_threadpool(svc.analyze, content, file.filename, gt)
    return await analysis_response(res, include_predictions, top_alerts)


@app.post("/analyze-groups")
async def analyze_groups(pressures: UploadFile = File(...), demands: UploadFile = File(...),
                         flows: UploadFile = File(...), levels: UploadFile = File(...),
                         ground_truth: UploadFile | None = File(None, description="Optional labels, evaluation only"),
                         include_predictions: bool = Query(True), top_alerts: int = Query(5, ge=0, le=20)):
    """Same as /analyze, for one file per sensor group (as BattLeDIM publishes them)."""
    files = {}
    for g, f in {"pressures": pressures, "demands": demands, "flows": flows, "levels": levels}.items():
        files[g] = (await read_upload(f), f.filename)
    gt = None
    if ground_truth is not None and ground_truth.filename:
        gt = await run_in_threadpool(svc.read_ground_truth, await read_upload(ground_truth), ground_truth.filename)
    wide = await run_in_threadpool(svc.combine_group_files, files)
    res = await run_in_threadpool(svc.analyze_frame, wide, gt)
    return await analysis_response(res, include_predictions, top_alerts)


async def analysis_response(res: svc.AnalysisResult, include_predictions: bool, top_alerts: int):
    if not res.ok:
        return error(422, "VALIDATION_FAILED", "The file did not pass validation.", validation=res.report.to_dict())
    top = res.results.nlargest(top_alerts, "risk_probability")
    top = top[top["alert"]]
    explanations = [await run_in_threadpool(res.explain, t) for t in top["Timestamp"]]
    return jsonable({
        "mode": "user_analysis",
        "validation": res.report.to_dict(),
        "summary": res.summary,
        "predictions": predictions_payload(res) if include_predictions else None,
        "top_alerts": explanations,
        "evaluation": res.evaluation,
        "notices": {k: WORDING[k] for k in ("not_live", "different_network", "inspection_guidance")},
    })


@app.post("/explain")
async def explain(file: UploadFile = File(...), timestamp: str = Form(...)):
    content = await read_upload(file)
    res = await run_in_threadpool(svc.analyze, content, file.filename, None)
    if not res.ok:
        return error(422, "VALIDATION_FAILED", "The file did not pass validation.", validation=res.report.to_dict())
    try:
        ts = pd.Timestamp(timestamp)
    except (ValueError, TypeError):
        return error(400, "BAD_TIMESTAMP", "Could not parse 'timestamp'.")
    try:
        return jsonable(await run_in_threadpool(res.explain, ts))
    except KeyError:
        return error(404, "TIMESTAMP_NOT_FOUND", "That timestamp is not in the analysed data.")


# ------------------------------------------------------------------ benchmark demo (pre-computed, read-only)
def _csv(name, **kw):
    return pd.read_csv(OUTPUTS_DIR / name, **kw)


@app.get("/benchmark")
def benchmark():
    pred = _csv("predictions_v2.csv")
    perm = _csv("permutation_importance_v2.csv")
    return jsonable({
        "mode": "benchmark_demo",
        "metrics": V2_VERIFIED | {"pr_auc": json.loads((OUTPUTS_DIR / "audit_report.json").read_text())["verified_test_metrics"]["pr_auc"],
                                  "threshold": 0.22, "n_test": int(len(pred)), "n_severe": int(pred["target"].sum()),
                                  "n_alerts": int(pred["prediction"].sum())},
        "v1": V1_VERIFIED,
        "splits": json.loads((OUTPUTS_DIR / "audit_report.json").read_text())["splits"],
        "episodes": _csv("event_summary_v2.csv").to_dict(orient="records"),
        "threshold_validation": _csv("threshold_validation_v2.csv").round(4).to_dict(orient="records"),
        "curves": {c: g[["x", "y"]].to_dict(orient="list") for c, g in _csv("curves_v2.csv").groupby("curve")},
        "baselines": _csv("experiments/baseline_comparison.csv").to_dict(orient="records"),
        "severity_sensitivity": _csv("experiments/severity_threshold_sensitivity.csv").to_dict(orient="records"),
        "feature_importance": perm[["feature", "mdi_importance", "validation_auc_drop_mean", "test_auc_drop_mean"]]
        .to_dict(orient="records"),
        "inspection_check": _csv("inspection_check_v2.csv").to_dict(orient="records"),
        "wording": WORDING,
    })


@app.get("/benchmark/timeline")
def benchmark_timeline(resolution: str = Query("1h", pattern="^(5min|1h)$")):
    p = _csv("predictions_v2.csv", parse_dates=["Timestamp"])
    if resolution == "1h":
        g = p.set_index("Timestamp").resample("1h")
        p = pd.DataFrame({"risk_probability": g["risk_probability"].max(), "alert": g["prediction"].max(),
                          "total_leak": g["total_leak"].mean(), "target": g["target"].max()}).reset_index()
    else:
        p = p.rename(columns={"prediction": "alert"})
    return jsonable({"resolution": resolution, "aggregation": "hourly max risk / any alert / mean leakage" if resolution == "1h" else "none",
                     "timestamp": p["Timestamp"].astype(str).tolist(), "risk_probability": p["risk_probability"].round(5).tolist(),
                     "alert": p["alert"].astype(int).tolist(),
                     "ground_truth_total_leak_m3h": p["total_leak"].round(3).tolist(),
                     "ground_truth_severe": p["target"].astype(int).tolist(),
                     "ground_truth_note": WORDING["ground_truth"]})


@app.get("/benchmark/explain")
def benchmark_explain(timestamp: str):
    ts = pd.Timestamp(timestamp)
    ctx = _csv("dashboard_context_v2.csv.gz", parse_dates=["Timestamp"])
    pdev = _csv("pressure_deviation_v2.csv.gz", parse_dates=["Timestamp"])
    pred = _csv("predictions_v2.csv", parse_dates=["Timestamp"])
    row, drow, prow = ctx[ctx["Timestamp"] == ts], pdev[pdev["Timestamp"] == ts], pred[pred["Timestamp"] == ts]
    if row.empty:
        return error(404, "TIMESTAMP_NOT_FOUND", "Timestamp outside the benchmark test period (13 Sep - 31 Dec 2018, 5-min steps).")
    b = svc.load_bundle()
    r, prow = row.iloc[0], prow.iloc[0]
    sig = []
    for f in b.ref["explain_signals"]:
        t = b.ref["explain_typical"][f]
        sig.append({"signal": f, "current_value": r.get(f), "typical_median": t["median"], "typical_q05": t["q05"],
                    "typical_q95": t["q95"], "risk_change_if_typical": r.get("sens_" + f),
                    "global_importance_rank": b.mdi_rank.get(f)})
    sig.sort(key=lambda s: -(s["risk_change_if_typical"] or 0))
    return jsonable({"timestamp": str(ts), "risk_probability": prow["risk_probability"], "alert": bool(prow["prediction"]),
                     "signals": sig, "inspection": svc.generate_inspection_guidance(drow.iloc[0]),
                     "ground_truth": {"total_leak_m3h": prow["total_leak"], "severe": bool(prow["target"]),
                                      "note": WORDING["ground_truth"]}})


@app.get("/network")
def network():
    return jsonable({"links": _csv("network_links.csv")[["id", "kind", "x0", "y0", "x1", "y1"]].to_dict(orient="list"),
                     "pressure_sensors": _csv("network_sensors.csv").to_dict(orient="records"),
                     "source": "L-TOWN.inp, BattLeDIM (CC BY 4.0). Benchmark node IDs, not real locations."})
