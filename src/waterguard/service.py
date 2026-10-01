"""WaterGuard inference service — the ONE place where new SCADA data becomes risk results.

Both the Streamlit app (app.py / ui/) and the HTTP API (api/main.py) call these
functions; neither contains model logic of its own.

Pipeline (no retraining, ever):

    read_table -> validate_input -> prepare_features -> run_inference
               -> explain_alert / generate_inspection_guidance -> summarize_results
               (optional, separate) evaluate_against_ground_truth

Design rules
- Ground truth is never needed and can never reach the model: only the 119 schema
  columns survive validation, and the feature list is checked against the saved one.
- Features are built by the SAME function used in training (features.build_features_v2),
  then ordered exactly as models/features_v2.joblib, imputed with the SAVED training
  imputer and scored by the SAVED Random Forest with the SAVED threshold (0.22).
- Uploads are parsed from memory only; nothing is written to disk or executed.
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import MODELS_DIR, OUTPUTS_DIR, SEVERE_THRESHOLD
from .data import EXPECTED_LEAK_LOCATIONS
from .evaluation import episodes, point_metrics
from .features import build_features_v2, feature_category
from .inspection import pressure_deviation as _pressure_deviation
from .schema import GROUP_PREFIX, GROUP_UNITS, SHEET_NAMES, expected_columns, scada_to_wide, wide_to_scada

# ------------------------------------------------------------------ limits
MAX_UPLOAD_BYTES = 100 * 1024 * 1024   # 100 MB
MAX_ROWS = 110_000                     # a little over one year at 5-minute resolution
MIN_ROWS = 12                          # one hour
ALLOWED_EXTENSIONS = (".csv", ".xlsx")
STEP = pd.Timedelta("5min")
CONTEXT_STEPS = 6                      # 30-minute change features need 6 earlier steps
OUT_OF_RANGE_CAUTION = 0.05            # share of readings outside the training range
OUT_OF_RANGE_INCOMPATIBLE = 0.25
GROUND_TRUTH_HINTS = re.compile(r"leak|target|severe|label|ground|truth", re.I)


class InputError(ValueError):
    """A problem with the uploaded file itself (format, size, parsing). Carries a stable code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


@dataclass
class Issue:
    level: str          # "error" | "warning" | "info"
    code: str
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ValidationReport:
    status: str = "ok"                  # "ok" | "warning" | "error"
    issues: list[Issue] = field(default_factory=list)
    n_rows: int = 0
    start: str | None = None
    end: str | None = None
    interval_minutes: float | None = None
    n_gaps: int = 0
    missing_steps_in_gaps: int = 0
    missing_cells: int = 0
    missing_share: float = 0.0
    missing_by_group: dict = field(default_factory=dict)
    columns_with_missing: dict = field(default_factory=dict)
    ignored_columns: list[str] = field(default_factory=list)
    ground_truth_like_columns: list[str] = field(default_factory=list)
    compatibility: dict = field(default_factory=dict)

    def add(self, level, code, message, **details):
        self.issues.append(Issue(level, code, message, details))
        if level == "error":
            self.status = "error"
        elif level == "warning" and self.status == "ok":
            self.status = "warning"

    @property
    def ok(self) -> bool:
        return self.status != "error"

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ model bundle
@dataclass
class ModelBundle:
    model: object
    imputer: object
    features: list[str]
    threshold: float
    ref: dict
    p_median: pd.DataFrame
    p_spread: pd.DataFrame
    mdi_rank: dict

    @property
    def schema(self) -> dict[str, list[str]]:
        return self.ref["schema"]

    @property
    def columns(self) -> list[str]:
        return expected_columns(self.schema)


@lru_cache(maxsize=2)
def load_bundle(models_dir: Path = MODELS_DIR) -> ModelBundle:
    """Load the saved V2 artifacts once per process."""
    models_dir = Path(models_dir)
    feats = list(joblib.load(models_dir / "features_v2.joblib"))
    ref = json.loads((models_dir / "inference_reference_v2.json").read_text())
    if ref["feature_order"] != feats:
        raise RuntimeError("inference_reference_v2.json does not match features_v2.joblib")
    model = joblib.load(models_dir / "waterguard_rf_v2.joblib")
    if model.n_features_in_ != len(feats):
        raise RuntimeError("model and feature list disagree")
    imp = pd.read_csv(OUTPUTS_DIR / "feature_importance_v2.csv")
    return ModelBundle(
        model=model,
        imputer=joblib.load(models_dir / "imputer_v2.joblib"),
        features=feats,
        threshold=float(joblib.load(models_dir / "threshold_v2.joblib")),
        ref=ref,
        p_median=pd.read_csv(models_dir / "pressure_typical_median_v2.csv", index_col="slot"),
        p_spread=pd.read_csv(models_dir / "pressure_typical_spread_v2.csv", index_col="slot"),
        mdi_rank={f: i + 1 for i, f in enumerate(imp["feature"])},
    )


def model_available(models_dir: Path = MODELS_DIR) -> bool:
    return (Path(models_dir) / "waterguard_rf_v2.joblib").exists()


# ------------------------------------------------------------------ reading
def _read_csv(content: bytes) -> pd.DataFrame:
    header = content[:4096].split(b"\n", 1)[0]
    # European-style CSV (like the BattLeDIM files): ';' separator and ',' decimals. Decide from the header,
    # because data rows with decimal commas would otherwise break a comma-separated parse.
    european = header.count(b";") > header.count(b",")
    try:
        if european:
            df = pd.read_csv(io.BytesIO(content), sep=";", decimal=",", low_memory=False)
            df = df.loc[:, ~(df.columns.astype(str).str.startswith("Unnamed") & df.isna().all())]  # trailing ';'
        else:
            df = pd.read_csv(io.BytesIO(content), low_memory=False)
    except (pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as exc:
        raise InputError("MALFORMED_CSV", f"The CSV file could not be parsed: {str(exc).splitlines()[0]}") from None
    if not isinstance(df.index, pd.RangeIndex):
        # pandas silently turns surplus leading fields into an index when rows are longer than the header
        raise InputError("MALFORMED_CSV", "The CSV rows have more fields than the header row.")
    return df


def _read_xlsx(content: bytes) -> pd.DataFrame:
    if not content.startswith(b"PK") or not zipfile.is_zipfile(io.BytesIO(content)):
        raise InputError("MALFORMED_XLSX", "The file has an .xlsx extension but is not a valid Excel workbook.")
    try:
        xls = pd.ExcelFile(io.BytesIO(content), engine="openpyxl")
        sheets = xls.sheet_names
        if all(name in sheets for name in SHEET_NAMES.values()):
            # BattLeDIM 4-sheet layout
            parts = {g: xls.parse(SHEET_NAMES[g]) for g in SHEET_NAMES}
            ts = [pd.to_datetime(p["Timestamp"], errors="coerce").reset_index(drop=True) for p in parts.values()]
            if len({len(t) for t in ts}) != 1 or not all(t.equals(ts[0]) for t in ts[1:]):
                raise InputError("SHEETS_NOT_ALIGNED", "The four sheets do not share identical timestamps row by row.")
            return scada_to_wide(parts)
        return xls.parse(sheets[0])
    except InputError:
        raise
    except Exception as exc:  # noqa: BLE001 - any openpyxl/zip error means an unreadable workbook
        raise InputError("MALFORMED_XLSX", f"The Excel workbook could not be read ({type(exc).__name__}).") from None


def read_table(content: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded CSV/XLSX from memory into one wide table. Raises InputError."""
    name = (filename or "").lower()
    if not name.endswith(ALLOWED_EXTENSIONS):
        raise InputError("UNSUPPORTED_FORMAT", "Unsupported file type. Upload a .csv or .xlsx file.")
    if len(content) == 0:
        raise InputError("EMPTY_FILE", "The file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise InputError("FILE_TOO_LARGE", f"The file is larger than {MAX_UPLOAD_BYTES // 2**20} MB.")
    df = _read_csv(content) if name.endswith(".csv") else _read_xlsx(content)
    if df.empty:
        raise InputError("EMPTY_FILE", "The file contains no data rows.")
    return df


def combine_group_files(files: dict[str, tuple[bytes, str]]) -> pd.DataFrame:
    """Merge one file per sensor group (as BattLeDIM publishes them, e.g. 2019_SCADA_Pressures.csv)
    into the wide prefixed table. ``files`` maps group ('pressures', 'demands', 'flows', 'levels')
    to (content, filename). Each file: Timestamp + unprefixed sensor columns (already-prefixed
    columns are also accepted). Rows are joined on Timestamp (outer join; gaps show up in validation)."""
    unknown = set(files) - set(GROUP_PREFIX)
    if unknown:
        raise InputError("UNKNOWN_GROUP", f"Unknown sensor group(s): {', '.join(sorted(unknown))}.")
    missing = [g for g in GROUP_PREFIX if g not in files]
    if missing:
        raise InputError("MISSING_GROUP_FILE", f"A file is needed for each sensor group; missing: {', '.join(missing)}.")
    merged = None
    for g, (content, name) in files.items():
        df = read_table(content, name)
        ts_col = next((c for c in df.columns if str(c).strip().lower() == "timestamp"), None)
        if ts_col is None:
            raise InputError("MISSING_TIMESTAMP", f"The {g} file ({name}) has no 'Timestamp' column.")
        df = df.rename(columns={ts_col: "Timestamp"})
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        if df["Timestamp"].isna().any():
            raise InputError("BAD_TIMESTAMP", f"The {g} file ({name}) has timestamps that could not be read.")
        if df["Timestamp"].duplicated().any():
            raise InputError("DUPLICATE_TIMESTAMPS", f"The {g} file ({name}) has duplicate timestamps.")
        p = GROUP_PREFIX[g]
        df.columns = ["Timestamp"] + [c if str(c).startswith(p) else p + str(c).strip() for c in df.columns[1:]]
        merged = df if merged is None else merged.merge(df, on="Timestamp", how="outer")
    return merged.sort_values("Timestamp").reset_index(drop=True)


# ------------------------------------------------------------------ validation
def validate_input(raw: pd.DataFrame, bundle: ModelBundle | None = None) -> tuple[ValidationReport, pd.DataFrame | None]:
    """Check an uploaded wide table against the V2 input schema.

    Returns (report, clean) where ``clean`` holds ONLY the Timestamp and the 119 schema
    columns (numeric, sorted, timezone-naive), or None if there are blocking errors.
    """
    bundle = bundle or load_bundle()
    rep = ValidationReport()
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # 1. timestamp column
    ts_col = next((c for c in df.columns if c.lower() == "timestamp"), None)
    if ts_col is None:
        rep.add("error", "MISSING_TIMESTAMP", "No 'Timestamp' column found.")
        return rep, None
    df = df.rename(columns={ts_col: "Timestamp"})
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    bad = int(ts.isna().sum())
    if bad:
        first = int(np.flatnonzero(ts.isna().to_numpy())[0])
        rep.add("error", "BAD_TIMESTAMP", f"{bad} timestamp value(s) could not be read as dates (first at data row {first + 1}).",
                count=bad, first_row=first + 1)
        return rep, None
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert(None)
        rep.add("warning", "TIMEZONE_REMOVED", "Timestamps had a time zone; they were converted to UTC and the zone was dropped.")
    df["Timestamp"] = ts

    # 2. columns
    expected = bundle.columns
    present = set(df.columns)
    missing = [c for c in expected if c not in present]
    if missing:
        by_group = {g: [c for c in missing if c.startswith(p)] for g, p in GROUP_PREFIX.items()}
        by_group = {g: v for g, v in by_group.items() if v}
        unprefixed = [c for c in missing if c[2:] in present]
        hint = (" Some columns appear without their prefix (e.g. 'n1' instead of 'P_n1'); see the schema."
                if unprefixed else "")
        rep.add("error", "MISSING_COLUMNS",
                f"{len(missing)} required sensor column(s) are missing "
                f"({', '.join(f'{g}: {len(v)}' for g, v in by_group.items())}).{hint}",
                missing=missing[:50], missing_count=len(missing))
    extras = [c for c in df.columns if c not in expected]
    gt_like = [c for c in extras if c in EXPECTED_LEAK_LOCATIONS or GROUND_TRUTH_HINTS.search(c)]
    other = [c for c in extras if c not in gt_like]
    rep.ignored_columns, rep.ground_truth_like_columns = other, gt_like
    if gt_like:
        rep.add("warning", "GROUND_TRUTH_COLUMNS_IGNORED",
                f"{len(gt_like)} column(s) look like leakage labels ({', '.join(gt_like[:6])}). They were removed "
                f"and are never used by the model. Upload labels separately to evaluate results.", columns=gt_like)
    if other:
        rep.add("warning", "EXTRA_COLUMNS", f"{len(other)} extra column(s) were ignored ({', '.join(other[:6])}"
                f"{'…' if len(other) > 6 else ''}).", columns=other[:50])
    if missing:
        return rep, None

    # 3. rows
    n = len(df)
    if n < MIN_ROWS:
        rep.add("error", "TOO_FEW_ROWS", f"Only {n} rows; at least {MIN_ROWS} (one hour at 5-minute steps) are needed.")
    if n > MAX_ROWS:
        rep.add("error", "TOO_MANY_ROWS", f"{n:,} rows exceeds the limit of {MAX_ROWS:,} (about one year).")

    # 4. duplicates / order
    dup = int(df["Timestamp"].duplicated().sum())
    if dup:
        ex = df.loc[df["Timestamp"].duplicated(keep=False), "Timestamp"].astype(str).unique()[:3].tolist()
        rep.add("error", "DUPLICATE_TIMESTAMPS", f"{dup} duplicate timestamp(s) (e.g. {', '.join(ex)}).", count=dup, examples=ex)
    if not df["Timestamp"].is_monotonic_increasing:
        rep.add("warning", "UNSORTED_TIMESTAMPS", "Rows were not in time order; they were sorted.")
        df = df.sort_values("Timestamp", kind="stable")
    df = df.reset_index(drop=True)

    # 5. sampling interval
    d = df["Timestamp"].diff().dropna()
    if len(d):
        med = d.median()
        rep.interval_minutes = med.total_seconds() / 60
        off_grid = int(((d % STEP) != pd.Timedelta(0)).sum())
        if med != STEP:
            rep.add("error", "INVALID_SAMPLING", f"Detected sampling interval is {rep.interval_minutes:g} minutes; "
                    f"the model requires 5-minute data (its change features span 5 and 30 minutes).")
        elif off_grid:
            rep.add("error", "IRREGULAR_SAMPLING", f"{off_grid} time step(s) are not whole multiples of 5 minutes.")
        else:
            gaps = d[d > STEP]
            rep.n_gaps = int(len(gaps))
            rep.missing_steps_in_gaps = int(((gaps // STEP) - 1).sum())
            if rep.n_gaps:
                rep.add("warning", "GAPS", f"{rep.n_gaps} gap(s) in the time series ({rep.missing_steps_in_gaps} missing "
                        f"5-minute steps). No predictions are made for missing steps; change features next to a gap "
                        f"are filled from training medians and flagged.")
    rep.n_rows = n
    rep.start, rep.end = str(df["Timestamp"].min()), str(df["Timestamp"].max())

    # 6. numeric values and missing data
    clean = df[expected].copy()
    sensor_cols = expected[1:]
    non_numeric = 0
    for c in sensor_cols:
        if not pd.api.types.is_numeric_dtype(clean[c]):
            conv = pd.to_numeric(clean[c], errors="coerce")
            non_numeric += int(conv.isna().sum() - clean[c].isna().sum())
            clean[c] = conv
    if non_numeric:
        rep.add("warning", "NON_NUMERIC_VALUES", f"{non_numeric} non-numeric value(s) were treated as missing.")
    na = clean[sensor_cols].isna()
    rep.missing_cells = int(na.to_numpy().sum())
    rep.missing_share = float(rep.missing_cells / na.size) if na.size else 0.0
    rep.missing_by_group = {g: float(na[[c for c in sensor_cols if c.startswith(p)]].to_numpy().mean())
                            for g, p in GROUP_PREFIX.items()}
    col_share = na.mean()
    rep.columns_with_missing = {c: round(float(v), 4) for c, v in col_share[col_share > 0].sort_values(ascending=False).head(20).items()}
    empty_cols = col_share[col_share == 1].index.tolist()
    if empty_cols:
        rep.add("error", "EMPTY_COLUMNS", f"{len(empty_cols)} required column(s) contain no values ({', '.join(empty_cols[:6])}).",
                columns=empty_cols)
    elif rep.missing_cells:
        level = "warning" if rep.missing_share > 0.05 else "info"
        rep.add(level, "MISSING_VALUES", f"{rep.missing_share:.2%} of sensor readings are missing. They are filled with the "
                f"model's training medians and the affected rows are flagged as lower confidence.")

    # 7. network compatibility: are values in the range the L-Town model was trained on?
    rng = bundle.ref["input_ranges_train"]
    lo = np.array([rng[c]["min"] for c in sensor_cols]); hi = np.array([rng[c]["max"] for c in sensor_cols])
    vals = clean[sensor_cols].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        outside = ((vals < lo) | (vals > hi)) & ~np.isnan(vals)
    share = float(outside.sum() / max(1, (~np.isnan(vals)).sum()))
    per_group = {g: float(outside[:, [i for i, c in enumerate(sensor_cols) if c.startswith(p)]].mean())
                 for g, p in GROUP_PREFIX.items()}
    if share <= OUT_OF_RANGE_CAUTION:
        level, msg = "compatible", "Schema matches L-Town and values lie within the range seen in training."
    elif share <= OUT_OF_RANGE_INCOMPATIBLE:
        level, msg = "caution", ("Schema matches L-Town, but a noticeable share of readings lies outside the training "
                                 "range. Treat results with caution.")
    else:
        level, msg = "out_of_distribution", ("Schema matches, but most readings are outside the range the model was "
                                             "trained on. This is probably a different network or operating regime; "
                                             "results are not meaningful without retraining.")
    rep.compatibility = {"schema": "L-Town V2 (119 sensor columns)", "level": level, "message": msg,
                         "share_outside_training_range": round(share, 4),
                         "share_outside_by_group": {k: round(v, 4) for k, v in per_group.items()}}
    if level != "compatible":
        rep.add("warning", "VALUES_OUTSIDE_TRAINING_RANGE", msg, share=round(share, 4))

    if not rep.ok:
        return rep, None
    return rep, clean.reset_index(drop=True)


# ------------------------------------------------------------------ features & inference
def prepare_features(clean: pd.DataFrame, bundle: ModelBundle | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the 60 V2 features with the training feature code, in the saved order.

    The data is first placed on a regular 5-minute grid so that 5- and 30-minute changes are
    never computed across a gap; grid rows that were not in the upload are dropped again.
    Returns (X, flags) aligned with ``clean`` rows. X is BEFORE imputation.
    """
    bundle = bundle or load_bundle()
    grid = pd.date_range(clean["Timestamp"].iloc[0], clean["Timestamp"].iloc[-1], freq=STEP)
    reg = clean.set_index("Timestamp").reindex(grid).rename_axis("Timestamp").reset_index()
    X = build_features_v2(wide_to_scada(reg, bundle.schema))
    X = X[X["Timestamp"].isin(clean["Timestamp"])].reset_index(drop=True)
    feats = [c for c in X.columns if c != "Timestamp"]
    if feats != bundle.features:  # guaranteed by construction; checked anyway
        X = X[["Timestamp"] + bundle.features]
    assert [c for c in X.columns if c != "Timestamp"] == bundle.features
    diff_cols = [c for c in bundle.features if "_diff_" in c]
    level_cols = [c for c in bundle.features if c not in diff_cols]
    flags = pd.DataFrame({
        "reduced_context": X[diff_cols].isna().any(axis=1).to_numpy(),
        "imputed_values": clean.drop(columns="Timestamp").isna().any(axis=1).to_numpy() | X[level_cols].isna().any(axis=1).to_numpy(),
    })
    return X, flags


def run_inference(X: pd.DataFrame, bundle: ModelBundle | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Saved imputer -> saved Random Forest -> saved threshold. Returns (results, imputed matrix)."""
    bundle = bundle or load_bundle()
    Xi = bundle.imputer.transform(X[bundle.features])
    prob = bundle.model.predict_proba(Xi)[:, 1]
    res = pd.DataFrame({"Timestamp": X["Timestamp"].to_numpy(), "risk_probability": prob,
                        "alert": prob >= bundle.threshold})
    return res, Xi


def explain_alert(x_imputed: np.ndarray, x_raw: pd.Series, bundle: ModelBundle | None = None) -> pd.DataFrame:
    """Local sensitivity for one observation: replace ONE key signal at a time with its typical
    (non-severe training median) value and re-score. Same method as the benchmark dashboard.
    Describes model behaviour, not physical cause."""
    bundle = bundle or load_bundle()
    signals, typ = bundle.ref["explain_signals"], bundle.ref["explain_typical"]
    idx = {f: i for i, f in enumerate(bundle.features)}
    base = bundle.model.predict_proba(x_imputed.reshape(1, -1))[:, 1][0]
    M = np.repeat(x_imputed.reshape(1, -1), len(signals), axis=0)
    for k, f in enumerate(signals):
        M[k, idx[f]] = typ[f]["median"]
    p_mod = bundle.model.predict_proba(M)[:, 1]
    rows = []
    for k, f in enumerate(signals):
        t = typ[f]
        cur = x_raw.get(f, np.nan)
        rows.append({
            "signal": f, "category": feature_category(f), "current_value": None if pd.isna(cur) else float(cur),
            "typical_median": t["median"], "typical_q05": t["q05"], "typical_q95": t["q95"],
            "deviation_sigma": None if pd.isna(cur) or not t["std"] else float((cur - t["median"]) / t["std"]),
            "outside_typical_range": None if pd.isna(cur) else bool(cur < t["q05"] or cur > t["q95"]),
            "risk_change_if_typical": float(base - p_mod[k]),
            "global_importance_rank": bundle.mdi_rank.get(f),
        })
    return pd.DataFrame(rows).sort_values("risk_change_if_typical", ascending=False).reset_index(drop=True)


def pressure_deviations(clean: pd.DataFrame, bundle: ModelBundle | None = None) -> pd.DataFrame:
    """Time-of-day-adjusted deviation of each pressure sensor (negative = lower than usual)."""
    bundle = bundle or load_bundle()
    p = clean[["Timestamp"] + [GROUP_PREFIX["pressures"] + s for s in bundle.schema["pressures"]]].copy()
    p.columns = ["Timestamp"] + bundle.schema["pressures"]
    dev = _pressure_deviation(p, p["Timestamp"], bundle.p_median, bundle.p_spread)
    return dev.reset_index(drop=True)


def generate_inspection_guidance(dev_row: pd.Series, top_n: int = 8) -> dict:
    """Rank pressure sensors by drop below their usual level. Inspection guidance, NOT localisation."""
    v = dev_row.drop(labels=["Timestamp"], errors="ignore").astype(float).dropna().sort_values()
    return {
        "ranking": [{"rank": i + 1, "sensor": s, "deviation_sigma": round(float(z), 3)} for i, (s, z) in enumerate(v.head(top_n).items())],
        "all_deviations": {s: round(float(z), 3) for s, z in v.items()},
        "sensors_below_minus_2_sigma": int((v < -2).sum()),
        "n_sensors": int(len(v)),
        "note": "Inspection guidance, not leak localisation: sensors furthest below their usual level for this time of day.",
    }


def summarize_results(results: pd.DataFrame, flags: pd.DataFrame | None = None) -> dict:
    p = results["risk_probability"].to_numpy()
    a = results["alert"].to_numpy().astype(bool)
    hist, edges = np.histogram(p, bins=np.linspace(0, 1, 11))
    eps = episodes(results["Timestamp"], pd.Series(a))
    i_max = int(np.argmax(p)) if len(p) else 0
    out = {
        "n_observations": int(len(results)),
        "start": str(results["Timestamp"].min()), "end": str(results["Timestamp"].max()),
        "n_alerts": int(a.sum()), "alert_share": float(a.mean()) if len(a) else 0.0,
        "max_risk": float(p.max()) if len(p) else None,
        "max_risk_timestamp": str(results["Timestamp"].iloc[i_max]) if len(p) else None,
        "mean_risk": float(p.mean()) if len(p) else None,
        "risk_histogram": {"bin_edges": [round(float(e), 2) for e in edges], "counts": hist.astype(int).tolist()},
        "alert_periods": [{"start": str(e.start), "end": str(e.end), "steps": int(e.n_steps),
                           "peak_risk": float(p[e.start_idx:e.end_idx + 1].max())} for _, e in eps.iterrows()],
    }
    if flags is not None:
        out["n_reduced_context"] = int(flags["reduced_context"].sum())
        out["n_imputed_values"] = int(flags["imputed_values"].sum())
    return out


# ------------------------------------------------------------------ optional ground truth (evaluation only)
def read_ground_truth(content: bytes, filename: str) -> pd.DataFrame:
    """Labels file: Timestamp + either 'total_leak' or one column per leak (summed). m³/h."""
    df = read_table(content, filename)
    ts_col = next((c for c in df.columns if str(c).lower() == "timestamp"), None)
    if ts_col is None:
        raise InputError("GT_MISSING_TIMESTAMP", "The ground-truth file has no 'Timestamp' column.")
    df = df.rename(columns={ts_col: "Timestamp"})
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    if "total_leak" in df.columns:
        total = pd.to_numeric(df["total_leak"], errors="coerce")
    else:
        num = df.drop(columns="Timestamp").apply(pd.to_numeric, errors="coerce")
        if num.shape[1] == 0:
            raise InputError("GT_NO_VALUES", "The ground-truth file has no leakage columns.")
        total = num.sum(axis=1, min_count=1)
    return pd.DataFrame({"Timestamp": df["Timestamp"], "total_leak": total}).dropna()


def evaluate_against_ground_truth(results: pd.DataFrame, gt: pd.DataFrame) -> dict:
    """Compare predictions with labels AFTER inference. Labels never touch features or the model."""
    m = results.merge(gt, on="Timestamp", how="inner")
    if m.empty:
        return {"matched_rows": 0, "note": "No ground-truth timestamps match the analysed data."}
    y = (m["total_leak"] >= SEVERE_THRESHOLD).astype(int)
    met = point_metrics(y, m["alert"].astype(int), m["risk_probability"])
    met = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in met.items()}
    met.update(matched_rows=int(len(m)), severe_rate=float(y.mean()), severe_threshold_m3h=SEVERE_THRESHOLD,
               note="Evaluation only: labels were joined after inference and never used by the model.")
    return met


# ------------------------------------------------------------------ orchestration
@dataclass
class AnalysisResult:
    report: ValidationReport
    clean: pd.DataFrame | None = None
    X: pd.DataFrame | None = None
    X_imputed: np.ndarray | None = None
    flags: pd.DataFrame | None = None
    results: pd.DataFrame | None = None
    deviations: pd.DataFrame | None = None
    summary: dict | None = None
    evaluation: dict | None = None
    ground_truth: pd.DataFrame | None = None

    @property
    def ok(self) -> bool:
        return self.results is not None

    def row_index(self, ts) -> int | None:
        hit = np.flatnonzero(self.results["Timestamp"].to_numpy() == np.datetime64(pd.Timestamp(ts)))
        return int(hit[0]) if len(hit) else None

    def explain(self, ts, bundle: ModelBundle | None = None) -> dict:
        i = self.row_index(ts)
        if i is None:
            raise KeyError("timestamp not in analysed data")
        r = self.results.iloc[i]
        return {
            "timestamp": str(r["Timestamp"]), "risk_probability": float(r["risk_probability"]), "alert": bool(r["alert"]),
            "reduced_context": bool(self.flags["reduced_context"].iloc[i]),
            "imputed_values": bool(self.flags["imputed_values"].iloc[i]),
            "signals": explain_alert(self.X_imputed[i], self.X.iloc[i], bundle).to_dict(orient="records"),
            "inspection": generate_inspection_guidance(self.deviations.iloc[i]),
        }

    def predictions_frame(self) -> pd.DataFrame:
        out = self.results.copy()
        out["alert"] = out["alert"].astype(int)
        out = pd.concat([out, self.flags.astype(int)], axis=1)
        if self.ground_truth is not None:  # evaluation columns, clearly named, joined AFTER inference
            g = self.ground_truth.rename(columns={"total_leak": "ground_truth_total_leak_m3h"})
            out = out.merge(g, on="Timestamp", how="left")
            out["ground_truth_severe"] = (out["ground_truth_total_leak_m3h"] >= SEVERE_THRESHOLD).astype("Int64").where(
                out["ground_truth_total_leak_m3h"].notna())
        return out


def analyze_frame(raw: pd.DataFrame, ground_truth: pd.DataFrame | None = None,
                  bundle: ModelBundle | None = None) -> AnalysisResult:
    """Full pipeline on an already-parsed wide table."""
    bundle = bundle or load_bundle()
    report, clean = validate_input(raw, bundle)
    if clean is None:
        return AnalysisResult(report=report)
    X, flags = prepare_features(clean, bundle)
    results, Xi = run_inference(X, bundle)
    res = AnalysisResult(report=report, clean=clean, X=X, X_imputed=Xi, flags=flags, results=results,
                         deviations=pressure_deviations(clean, bundle), summary=summarize_results(results, flags))
    if ground_truth is not None:
        res.ground_truth = ground_truth
        res.evaluation = evaluate_against_ground_truth(results, ground_truth)
    return res


def analyze(content: bytes, filename: str, ground_truth: tuple[bytes, str] | None = None,
            bundle: ModelBundle | None = None) -> AnalysisResult:
    """Full pipeline from uploaded bytes. Raises InputError for unreadable files."""
    raw = read_table(content, filename)
    gt = read_ground_truth(*ground_truth) if ground_truth else None
    return analyze_frame(raw, gt, bundle)


def schema_description(bundle: ModelBundle | None = None) -> dict:
    bundle = bundle or load_bundle()
    return {
        "format": "One table, one row per 5-minute timestamp. CSV (comma-separated, or ';' with decimal commas) or XLSX "
                  "(one sheet in this wide format, or the BattLeDIM 4-sheet layout).",
        "timestamp_column": "Timestamp",
        "sampling_interval_minutes": 5,
        "column_groups": {g: {"prefix": GROUP_PREFIX[g], "unit": GROUP_UNITS[g], "count": len(bundle.schema[g]),
                              "columns": [GROUP_PREFIX[g] + c for c in bundle.schema[g]]} for g in GROUP_PREFIX},
        "total_sensor_columns": len(bundle.columns) - 1,
        "ground_truth": "Not required and never used for prediction. Optionally upload labels as a separate file "
                        "(Timestamp + total_leak, or one column per leak, m³/h) for evaluation only.",
        "limits": {"max_file_mb": MAX_UPLOAD_BYTES // 2**20, "max_rows": MAX_ROWS, "min_rows": MIN_ROWS,
                   "allowed_extensions": list(ALLOWED_EXTENSIONS)},
        "network": "Columns must be the L-Town benchmark sensors. Data from a different network is not compatible "
                   "with this model without retraining.",
    }


def template_csv(bundle: ModelBundle | None = None) -> bytes:
    bundle = bundle or load_bundle()
    return (",".join(bundle.columns) + "\n").encode()
