"""2. Analyze Data — upload -> validate -> features -> saved V2 model -> results -> export.
All processing is done by waterguard.service; this page only collects input and displays output."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.common import (C, USER, alert_detail, bundle, cards, fmt_ts, hero, issue_list, link_to, note, risk_timeline,
                       section, show, style, user_view)
from waterguard import service as svc
from waterguard.schema import GROUP_PREFIX, GROUP_UNITS
from waterguard.wording import WORDING

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "samples" / "waterguard_sample_ltown2018_shifted_to_2026.csv"
SAMPLE_GT = ROOT / "samples" / "waterguard_sample_ground_truth_shifted_to_2026.csv"
TEMPLATE = ROOT / "samples" / "waterguard_input_template.csv"
COMPAT = {"compatible": ("Compatible", "good"), "caution": ("Use with caution", "caution"),
          "out_of_distribution": ("Not compatible", "warn")}


def _set_source(name, content, origin, frame=None):
    st.session_state["source"] = {"name": name, "bytes": content, "origin": origin, "frame": frame,
                                  "hash": hashlib.sha256(content).hexdigest()}


def _parsed():
    """Read + validate the current source once per file (kept in session state)."""
    src = st.session_state.get("source")
    if not src:
        return None
    cache = st.session_state.get("parsed")
    if cache and cache["hash"] == src["hash"]:
        return cache
    try:
        raw = src["frame"] if src.get("frame") is not None else svc.read_table(src["bytes"], src["name"])
        report, clean = svc.validate_input(raw, bundle())
        cache = {"hash": src["hash"], "raw": raw, "report": report, "clean": clean, "error": None}
    except svc.InputError as e:
        cache = {"hash": src["hash"], "raw": None, "report": None, "clean": None, "error": e}
    st.session_state["parsed"] = cache
    return cache


def _schema_expander():
    with st.expander("Expected input schema (119 sensor columns + Timestamp)"):
        sch = svc.schema_description(bundle())
        st.markdown(f"{sch['format']} Timestamps every **5 minutes**. Any year is accepted. "
                    f"Limits: {sch['limits']['max_file_mb']} MB, {sch['limits']['max_rows']:,} rows, at least "
                    f"{sch['limits']['min_rows']} rows.")
        st.dataframe(pd.DataFrame([{"Group": g.title(), "Prefix": v["prefix"], "Columns": v["count"], "Unit": v["unit"],
                                    "Examples": ", ".join(v["columns"][:4]) + ("…" if v["count"] > 4 else "")}
                                   for g, v in sch["column_groups"].items()]), width="stretch", hide_index=True)
        st.caption("Prefixes are needed because some L-Town nodes carry both a pressure sensor and a demand meter "
                   "(e.g. P_n1 and D_n1). Leakage labels must NOT be in this file; upload them separately if you have them.")
        st.code(", ".join(bundle().columns), language=None, wrap_lines=True)


def _validation_panel(parsed, src):
    if parsed["error"] is not None:
        e = parsed["error"]
        note(f"<b>The file could not be read</b> ({e.code}). {e.message}", "err")
        return False
    rep: svc.ValidationReport = parsed["report"]
    status = {"ok": ("Ready to analyse", "good"), "warning": ("Ready, with warnings", "caution"),
              "error": ("Cannot be analysed", "warn")}[rep.status]
    compat = COMPAT.get(rep.compatibility.get("level"), ("Not checked", ""))
    cards([
        ("Status", status[0], f"{src['name']}", status[1]),
        ("Observations", f"{rep.n_rows:,}" if rep.n_rows else "—", "rows (5-minute steps)", ""),
        ("Date range", (f"{pd.Timestamp(rep.start):%d %b %Y}" if rep.start else "—"),
         (f"to {pd.Timestamp(rep.end):%d %b %Y %H:%M}" if rep.end else ""), ""),
        ("Sampling interval", f"{rep.interval_minutes:g} min" if rep.interval_minutes else "—",
         f"{rep.n_gaps} gap(s)" if rep.interval_minutes else "", ""),
        ("Missing readings", f"{rep.missing_share:.2%}" if rep.n_rows else "—", f"{rep.missing_cells:,} cells", ""),
        ("Network compatibility", compat[0],
         f"{rep.compatibility.get('share_outside_training_range', 0):.1%} outside training range" if rep.compatibility else "", compat[1]),
    ])
    section("Validation findings")
    issue_list(rep.issues)
    if rep.compatibility:
        st.caption(rep.compatibility["message"])
    if parsed["clean"] is not None:
        with st.expander("Preview and missing-data summary"):
            clean = parsed["clean"]
            cols = ["Timestamp"] + [c for c in clean.columns if c.startswith(("P_n1", "P_n229", "F_", "L_"))][:8]
            st.dataframe(clean[cols].head(12), width="stretch", hide_index=True)
            st.caption(f"Showing 12 of {len(clean):,} rows and {len(cols) - 1} of 119 sensor columns.")
            st.dataframe(pd.DataFrame([{"Group": g.title(), "Unit": GROUP_UNITS[g], "Missing share": f"{v:.2%}"}
                                       for g, v in rep.missing_by_group.items()]), width="stretch", hide_index=True)
    return rep.ok


def _results(res: svc.AnalysisResult, filename: str):
    view = user_view()
    s = res.summary
    cards([
        ("Observations", f"{s['n_observations']:,}", f"{pd.Timestamp(s['start']):%d %b %Y} – {pd.Timestamp(s['end']):%d %b %Y}", ""),
        ("Alerts", f"{s['n_alerts']:,}", f"{s['alert_share']:.1%} of observations", "warn" if s["n_alerts"] else "good"),
        ("Highest risk", f"{s['max_risk']:.3f}", fmt_ts(s["max_risk_timestamp"]), "accent"),
        ("Alert periods", f"{len(s['alert_periods'])}", "contiguous runs of alerts", ""),
        ("Lower-confidence rows", f"{s['n_reduced_context'] + s['n_imputed_values']:,}",
         f"{s['n_reduced_context']} reduced context · {s['n_imputed_values']} imputed", ""),
    ])
    lvl = res.report.compatibility.get("level")
    if lvl != "compatible":
        note(f"<b>Network compatibility: {COMPAT[lvl][0]}.</b> {res.report.compatibility['message']}", "err")

    section("Risk timeline")
    fig = risk_timeline(view, 360)
    al = view.pred[view.pred["alert"] == 1]
    fig.add_trace(go.Scattergl(x=al["Timestamp"], y=al["risk_probability"], mode="markers", name="Alert",
                               marker=dict(color=C["alert"], size=4)))
    show(fig)
    if view.has_truth:
        note(WORDING["ground_truth"] + " The labels you uploaded were joined to the predictions after inference.", "gt")

    c1, c2 = st.columns([1, 1.3])
    with c1:
        section("Risk distribution")
        h = s["risk_histogram"]
        e = h["bin_edges"]
        fig = go.Figure(go.Bar(x=[f"{e[i]:.1f}–{e[i + 1]:.1f}" for i in range(len(e) - 1)], y=h["counts"],
                               marker_color=[C["alert"] if e[i] >= 0.2 else C["sky"] for i in range(len(e) - 1)],
                               hovertemplate="risk %{x}: %{y} observations<extra></extra>"))
        fig.update_xaxes(title="Risk probability")
        fig.update_yaxes(title="Observations")
        show(style(fig, 300, legend=False))
        st.caption("Red bars contain the 0.22 alert threshold or lie above it.")
    with c2:
        section("Alert periods")
        if s["alert_periods"]:
            ap = pd.DataFrame(s["alert_periods"])
            st.dataframe(pd.DataFrame({"Start": pd.to_datetime(ap["start"]).dt.strftime("%d %b %Y %H:%M"),
                                       "End": pd.to_datetime(ap["end"]).dt.strftime("%d %b %Y %H:%M"),
                                       "Duration (h)": (ap["steps"] * 5 / 60).round(2), "Peak risk": ap["peak_risk"].round(3)})
                         .sort_values("Peak risk", ascending=False), width="stretch", hide_index=True, height=300)
        else:
            note("No alerts were raised for this data.")

    if res.evaluation:
        section("Evaluation against your labels (evaluation only)")
        ev = res.evaluation
        if ev.get("matched_rows"):
            cards([("Matched rows", f"{ev['matched_rows']:,}", "label timestamps found", ""),
                   ("Precision", f"{ev['precision']:.1%}", "alerts that were severe", "good"),
                   ("Recall", f"{ev['recall']:.1%}", "severe steps alerted", "warn"),
                   ("Severe share", f"{ev['severe_rate']:.1%}", "total leakage ≥ 40 m³/h", "")])
            note(ev["note"], "gt")
        else:
            note(ev.get("note", "No matching labels."), "gt")

    section("Alerts — select one to see why it was flagged")
    top = view.pred[view.pred["alert"] == 1].nlargest(200, "risk_probability")
    if top.empty:
        note("No alerts to explain. The Risk & Alerts page lets you inspect any timestamp.")
    else:
        tbl = pd.DataFrame({"Timestamp": top["Timestamp"].dt.strftime("%Y-%m-%d %H:%M"),
                            "Risk probability": top["risk_probability"].round(3)})
        fl = res.flags.set_index(res.results["Timestamp"]).loc[top["Timestamp"]]
        tbl["Confidence"] = np.where(fl["reduced_context"].to_numpy() | fl["imputed_values"].to_numpy(), "Lower", "Normal")
        ev = st.dataframe(tbl, width="stretch", hide_index=True, height=260, on_select="rerun",
                          selection_mode="single-row", key="analyze_alert_table",
                          column_config={"Risk probability": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1)})
        sel = ev.selection.rows[0] if ev and ev.selection.rows else 0
        ts = top["Timestamp"].iloc[sel]
        with st.container(border=True):
            st.markdown(f"**Alert detail — {fmt_ts(ts)}**")
            alert_detail(view, ts)
            g = svc.generate_inspection_guidance(res.deviations.iloc[res.row_index(ts)])
            section("Inspection guidance — sensors furthest below usual")
            st.dataframe(pd.DataFrame(g["ranking"]).rename(columns={"rank": "Rank", "sensor": "Pressure sensor",
                                                                    "deviation_sigma": "Deviation (σ)"}),
                         width="stretch", hide_index=True)
            st.caption(WORDING["inspection_guidance"])

    section("Export")
    out = res.predictions_frame()
    d1, d2, d3 = st.columns(3)
    stem = Path(filename).stem
    d1.download_button("Predictions (CSV)", out.to_csv(index=False).encode(), file_name=f"{stem}_waterguard_predictions.csv",
                       mime="text/csv", icon=":material/download:", width="stretch")
    d2.download_button("Summary + validation (JSON)",
                       json.dumps({"summary": s, "validation": res.report.to_dict(), "evaluation": res.evaluation,
                                   "model": "WaterGuard V2, threshold 0.22", "notice": WORDING["different_network"]},
                                  indent=2, default=str).encode(),
                       file_name=f"{stem}_waterguard_summary.json", mime="application/json", icon=":material/download:",
                       width="stretch")
    with d3:
        link_to("alerts", "Open in Risk & Alerts", ":material/monitoring:")
        link_to("inspect", "Open in Inspection & Sensors", ":material/travel_explore:")
    st.caption("Predictions columns: Timestamp, risk_probability, alert (1 = risk ≥ 0.22), reduced_context, imputed_values"
               + (", ground_truth_total_leak_m3h, ground_truth_severe (evaluation only)" if res.ground_truth is not None else "") + ".")


def _group_uploads():
    st.markdown("Upload one file per sensor group, each with a `Timestamp` column and the sensor IDs as column names "
                "(e.g. `n1, n4, …` for pressures — prefixes are added automatically). Comma- or semicolon-separated "
                "CSV (decimal commas allowed) or XLSX. The BattLeDIM files `*_SCADA_Pressures.csv`, `*_SCADA_Demands.csv`, "
                "`*_SCADA_Flows.csv` and `*_SCADA_Levels.csv` work as they are.")
    cols = st.columns(4)
    ups = {g: cols[i].file_uploader(f"{g.title()} ({GROUP_UNITS[g]})", type=["csv", "xlsx"], key=f"grp_{g}")
           for i, g in enumerate(GROUP_PREFIX)}
    if not all(ups.values()):
        st.caption(f"{sum(u is not None for u in ups.values())} of 4 files provided.")
        return
    key = "|".join(u.file_id for u in ups.values())
    if st.session_state.get("group_id") != key:
        st.session_state["group_id"], st.session_state["group_error"] = key, None
        try:
            with st.spinner("Combining files…"):
                wide = svc.combine_group_files({g: (u.getvalue(), u.name) for g, u in ups.items()})
            _set_source("combined: " + ", ".join(u.name for u in ups.values()), key.encode(), "groups", frame=wide)
        except svc.InputError as e:
            st.session_state.pop("source", None)
            st.session_state["group_error"] = f"<b>The files could not be combined</b> ({e.code}). {e.message}"
    if st.session_state.get("group_error"):
        note(st.session_state["group_error"], "err")


def render():
    hero("Analyze Data", "Upload L-Town-compatible SCADA data and run the saved WaterGuard V2 model. No labels needed; "
         "no retraining.", badge=("user", "Analysis mode · saved V2 model · threshold 0.22"))
    note(f"<b>Only for L-Town-compatible data.</b> {WORDING['different_network']} {WORDING['time_recency']}", "gt")

    section("1 · Provide data")
    method = st.segmented_control("Input format", ["One file", "Separate files per sensor group"], default="One file",
                                  key="input_method",
                                  help="One wide CSV/XLSX in the WaterGuard schema (or the BattLeDIM 4-sheet workbook), or "
                                       "four files — pressures, demands, flows, levels — as BattLeDIM publishes them.") or "One file"
    if method == "Separate files per sensor group":
        _group_uploads()
    c1, c2 = st.columns([1.5, 1])
    with c1:
        up = None if method != "One file" else st.file_uploader("SCADA file (.csv or .xlsx)", type=["csv", "xlsx"], key="upload",
                              help="One row per 5-minute timestamp, 119 prefixed sensor columns. See the schema below.")
        if up is not None and st.session_state.get("upload_id") != up.file_id:
            st.session_state["upload_id"] = up.file_id
            _set_source(up.name, up.getvalue(), "upload")
        if st.button("Use the sample file", icon=":material/science:",
                     help="BattLeDIM L-Town 2018 measurements (4–7 Oct) with timestamps shifted to 2026."):
            _set_source(SAMPLE.name, SAMPLE.read_bytes(), "sample")
        st.caption("Sample = real BattLeDIM L-Town measurements from 4–7 Oct 2018 with timestamps shifted to 2026. "
                   "Not 2026 data, not Saudi data.")
    with c2:
        st.download_button("Blank template (CSV)", TEMPLATE.read_bytes(), file_name=TEMPLATE.name, mime="text/csv",
                           icon=":material/description:", width="stretch")
        st.download_button("Sample data (CSV)", SAMPLE.read_bytes(), file_name=SAMPLE.name, mime="text/csv",
                           icon=":material/download:", width="stretch")
        st.download_button("Sample labels (CSV, evaluation only)", SAMPLE_GT.read_bytes(), file_name=SAMPLE_GT.name,
                           mime="text/csv", icon=":material/download:", width="stretch")
    _schema_expander()
    with st.expander("Optional: ground-truth labels, for evaluation only"):
        st.markdown("If you know the true leakage for the same timestamps (`Timestamp` + `total_leak` in m³/h, or one "
                    "column per leak), upload it here. It is joined to the predictions **after** inference and is "
                    "**never** used by the model.")
        gt_up = st.file_uploader("Labels file (.csv or .xlsx)", type=["csv", "xlsx"], key="gt_upload")
        use_sample_gt = st.checkbox("Use the sample labels (only meaningful with the sample file)", key="use_sample_gt")

    src = st.session_state.get("source")
    if not src:
        note("Upload a file or click <b>Use the sample file</b> to begin. Nothing is stored: files are processed in "
             "memory for this session only.")
        return

    section("2 · Validate")
    parsed = _parsed()
    ok = _validation_panel(parsed, src)

    section("3 · Run analysis")
    run = st.button("Run analysis", type="primary", disabled=not ok, icon=":material/play_arrow:")
    if run:
        gt = None
        try:
            if gt_up is not None:
                gt = svc.read_ground_truth(gt_up.getvalue(), gt_up.name)
            elif use_sample_gt:
                gt = svc.read_ground_truth(SAMPLE_GT.read_bytes(), SAMPLE_GT.name)
        except svc.InputError as e:
            note(f"<b>Labels file ignored</b> ({e.code}). {e.message}", "err")
        with st.spinner("Building features and running the saved model…"):
            res = svc.analyze_frame(parsed["raw"], gt, bundle())
        st.session_state["analysis"] = {"result": res, "filename": src["name"], "hash": src["hash"]}
        st.session_state["mode"] = USER

    a = st.session_state.get("analysis")
    if not a:
        st.caption("The model has not been run on this file yet.")
        return
    if a["hash"] != src["hash"]:
        note(f"Showing results for the previously analysed file <b>{a['filename']}</b>. Click <b>Run analysis</b> to "
             f"analyse <b>{src['name']}</b>.")
    section(f"4 · Results — {a['filename']}")
    _results(a["result"], a["filename"])
