"""3. Risk & Alerts — monitor, detection timeline and alert explanation in one place."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui.common import (C, alert_detail, cards, empty_user_state, fmt_ts, hero, mode_selector, note, risk_timeline,
                       section, show, style)
from waterguard.config import SEVERE_THRESHOLD, V2_ALERT_THRESHOLD
from waterguard.evaluation import point_metrics
from waterguard.wording import WORDING


def _monitor(view):
    p = view.pred
    lo_d, hi_d = p["Timestamp"].min().date(), p["Timestamp"].max().date()
    c1, c2 = st.columns([2, 1])
    with c1:
        rng = st.date_input("Date range", value=(lo_d, hi_d), min_value=lo_d, max_value=hi_d, key=f"rng_{view.kind}")
    with c2:
        thr = st.slider("What-if alert threshold", 0.05, 0.80, V2_ALERT_THRESHOLD, 0.01, key=f"thr_{view.kind}",
                        help="The model's threshold is 0.22, chosen on validation data. Changing it here is exploratory "
                             "only; choosing a threshold by looking at test results would be a form of data leakage.")
    if not isinstance(rng, (tuple, list)) or len(rng) != 2:
        st.info("Select a start and an end date.")
        return
    lo, hi = pd.Timestamp(rng[0]), pd.Timestamp(rng[1]) + pd.Timedelta("1D")
    w = p[(p["Timestamp"] >= lo) & (p["Timestamp"] < hi)].copy()
    if w.empty:
        st.warning("No observations in this range.")
        return
    w["alert_w"] = (w["risk_probability"] >= thr).astype(int)
    if abs(thr - V2_ALERT_THRESHOLD) > 1e-9:
        note(f"Exploratory threshold {thr:.2f}. The model's alerts use {V2_ALERT_THRESHOLD:.2f}.", "gt")
    items = [
        ("Observations", f"{len(w):,}", f"{w['Timestamp'].min():%d %b} – {w['Timestamp'].max():%d %b %Y}", ""),
        ("Alerts", f"{int(w['alert_w'].sum()):,}", f"{w['alert_w'].mean():.1%} of window", "warn"),
        ("Peak risk", f"{w['risk_probability'].max():.2f}", fmt_ts(w.loc[w['risk_probability'].idxmax(), 'Timestamp']), "accent"),
    ]
    if view.has_truth and w["target"].notna().any():
        wt = w[w["target"].notna()]
        m = point_metrics(wt["target"].astype(int), wt["alert_w"])
        items.append(("Alerts confirmed severe", f"{m['tp']:,}" if m["tp"] + m["fp"] else "—",
                      f"precision {m['precision']:.1%} · evaluation only" if m["tp"] + m["fp"] else "no alerts", "good"))
    cards(items)
    show(risk_timeline(view, 300, thr, data=w))

    section("Alerts in window — select one to see why it was flagged")
    top = w[w["alert_w"] == 1].nlargest(200, "risk_probability")
    if top.empty:
        note("No alerts at this threshold in this window.")
        return
    tbl = pd.DataFrame({"Timestamp": top["Timestamp"].dt.strftime("%Y-%m-%d %H:%M"),
                        "Risk probability": top["risk_probability"].round(3)})
    if view.has_truth:
        tbl["Ground truth: severe? (evaluation only)"] = np.where(top["target"] == 1, "Yes", np.where(top["target"] == 0, "No", "—"))
    ev = st.dataframe(tbl, width="stretch", hide_index=True, height=280, on_select="rerun", selection_mode="single-row",
                      key=f"alerts_tbl_{view.kind}",
                      column_config={"Risk probability": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1)})
    sel = ev.selection.rows[0] if ev and ev.selection.rows else 0
    export = w[w["alert_w"] == 1].drop(columns=["alert_w"])
    st.download_button("Download alerts in window (CSV)", export.to_csv(index=False).encode(),
                       file_name=f"waterguard_{view.kind}_alerts_{rng[0]}_{rng[1]}_thr{thr:.2f}.csv", mime="text/csv",
                       icon=":material/download:")
    ts = top["Timestamp"].iloc[sel]
    with st.container(border=True):
        st.markdown(f"**Alert detail — {fmt_ts(ts)}**")
        alert_detail(view, ts)


def _timeline(view):
    p = view.pred
    if view.has_truth:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07, row_heights=[0.55, 0.45],
                            subplot_titles=("Ground truth: total leakage (evaluation only)", "Model output: risk probability"))
        fig.add_trace(go.Scattergl(x=p["Timestamp"], y=p["total_leak"], mode="lines", name="Total leakage (ground truth)",
                                   line=dict(color="#475569", width=1)), 1, 1)
        al = p[p["alert"] == 1]
        fig.add_trace(go.Scattergl(x=al["Timestamp"], y=al["total_leak"], mode="markers", name="Alert",
                                   marker=dict(color=C["alert"], size=3)), 1, 1)
        fig.add_hline(y=SEVERE_THRESHOLD, line_dash="dash", line_color=C["amber"], row=1, col=1,
                      annotation_text=f"experimental severity threshold = {SEVERE_THRESHOLD:.0f} m³/h", annotation_position="top left")
        fig.add_trace(go.Scattergl(x=p["Timestamp"], y=p["risk_probability"], mode="lines", name="Risk probability",
                                   line=dict(color=C["blue"], width=1)), 2, 1)
        fig.add_hline(y=V2_ALERT_THRESHOLD, line_dash="dash", line_color=C["alert"], row=2, col=1,
                      annotation_text=f"alert threshold {V2_ALERT_THRESHOLD:.2f}", annotation_position="top left")
        fig.update_yaxes(title="Leakage (m³/h)", row=1, col=1)
        fig.update_yaxes(title="Probability", range=[0, 1], row=2, col=1)
        style(fig, 620)
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0), margin=dict(t=40))
        show(fig)
        c1, c2 = st.columns(2)
        c1.html('<div class="wg-note"><b>Top</b>: recorded leakage. Values above the amber line count as severe in this '
                'experiment. Red dots: times the model raised an alert.</div>')
        c2.html('<div class="wg-note"><b>Bottom</b>: what the model produced from SCADA data alone. An alert is raised when '
                'the blue line reaches the red dashed threshold.</div>')
        if view.kind == "benchmark":
            note("Background leakage of roughly 18–25 m³/h is present almost all year (persistent small leaks), which is "
                 "why \"any leakage > 0\" was not a useful target.", "gt")
    else:
        fig = risk_timeline(view, 420)
        al = p[p["alert"] == 1]
        fig.add_trace(go.Scattergl(x=al["Timestamp"], y=al["risk_probability"], mode="markers", name="Alert",
                                   marker=dict(color=C["alert"], size=4)))
        show(fig)
        note("No labels were provided for this data, so only the model output is shown. Upload labels on Analyze Data "
             "(evaluation only) to compare alerts with known leakage.")
    st.caption("Drag across a chart to zoom; double-click to reset.")


def render():
    view = mode_selector("alerts")
    hero("Risk & Alerts", "Monitor risk over time, review alerts and see why each one was raised.",
         badge=view.badge if view else ("user", "Your analysis · no data yet"))
    if view is None:
        empty_user_state()
        return
    part = st.segmented_control("View", ["Monitor & alerts", "Detection timeline"], default="Monitor & alerts",
                                key=f"alerts_view_{view.kind}") or "Monitor & alerts"
    if part == "Monitor & alerts":
        _monitor(view)
    else:
        _timeline(view)
    if view.kind == "benchmark":
        st.caption(WORDING["ground_truth"])
