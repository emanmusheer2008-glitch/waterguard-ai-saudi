"""WaterGuard AI Saudi — decision-support dashboard (retrospective benchmark replay).

Reads only pre-computed files in outputs/. The model is evaluated on the
BattLeDIM L-Town international benchmark, NOT Saudi utility data.
Run:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from waterguard.config import SEVERE_THRESHOLD, V2_ALERT_THRESHOLD, V2_VERIFIED  # noqa: E402
from waterguard.dashboard_data import load_all, missing_outputs  # noqa: E402
from waterguard.evaluation import episodes, point_metrics  # noqa: E402
from waterguard.features import feature_category  # noqa: E402

st.set_page_config(page_title="WaterGuard AI Saudi", page_icon=":material/water_drop:", layout="wide")

# ------------------------------------------------------------------ palette
C = {
    "ink": "#0F1B2D", "muted": "#5B6B7F", "line": "#E3E8EF", "bg": "#F5F7FA",
    "blue": "#0B5FA5", "teal": "#0E8A83", "sky": "#7FB8E6",
    "alert": "#C8423B", "alert_bg": "rgba(200,66,59,0.10)", "amber": "#C98A13", "ok": "#2E7D5B",
}
CAT_COLORS = {
    "Individual pressure sensor": "#0B5FA5", "Network pressure summary": "#3E86C6",
    "Individual flow / pump": "#0E8A83", "Network flow summary": "#43B2A9",
    "Short-term change": "#8A6BBE", "Tank level": "#C98A13",
    "Time of day / calendar": "#8C98A8", "Demand summary": "#B0708F", "Other": "#B8C2CE",
}

st.html(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');
html, body, .stApp, .stMarkdown p, .stMarkdown li, h1, h2, h3, h4, label, input, textarea {{ font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; }}
.block-container {{ padding-top: 4.2rem; max-width: 1320px; }}
h1, h2, h3 {{ color: {C['ink']}; letter-spacing: -0.01em; }}
.wg-hero {{ display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; flex-wrap:wrap;
           border-bottom:1px solid {C['line']}; padding-bottom:.9rem; margin-bottom:1.1rem; }}
.wg-title {{ font-size:1.65rem; font-weight:700; color:{C['ink']}; line-height:1.15; }}
.wg-sub {{ color:{C['muted']}; font-size:.95rem; margin-top:.25rem; max-width:760px; }}
.wg-status {{ display:inline-flex; align-items:center; gap:.45rem; font-size:.78rem; font-weight:600;
             color:{C['blue']}; background:#EAF2FB; border:1px solid #CFE0F3; border-radius:999px; padding:.3rem .75rem; }}
.wg-dot {{ width:.5rem; height:.5rem; border-radius:50%; background:{C['blue']}; }}
.wg-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:.75rem; margin:.2rem 0 1rem; }}
.wg-card {{ background:#fff; border:1px solid {C['line']}; border-radius:10px; padding:.85rem 1rem; }}
.wg-card .k {{ font-size:.74rem; text-transform:uppercase; letter-spacing:.06em; color:{C['muted']}; font-weight:600; }}
.wg-card .v {{ font-family:'IBM Plex Mono', monospace; font-size:1.55rem; color:{C['ink']}; margin-top:.15rem; }}
.wg-card .n {{ font-size:.78rem; color:{C['muted']}; margin-top:.1rem; }}
.wg-card.accent {{ border-top:3px solid {C['blue']}; }}
.wg-card.warn {{ border-top:3px solid {C['alert']}; }}
.wg-card.good {{ border-top:3px solid {C['ok']}; }}
.wg-note {{ background:{C['bg']}; border-left:3px solid {C['sky']}; padding:.65rem .9rem; border-radius:4px;
           color:#33475B; font-size:.88rem; margin:.4rem 0 .8rem; }}
.wg-gt {{ background:#FFF8EC; border-left:3px solid {C['amber']}; padding:.6rem .9rem; border-radius:4px;
         color:#5A4410; font-size:.86rem; margin:.4rem 0 .8rem; }}
.wg-sec {{ font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; color:{C['muted']};
          font-weight:600; margin:1.2rem 0 .4rem; }}
.wg-pill {{ display:inline-block; padding:.12rem .55rem; border-radius:999px; font-size:.75rem; font-weight:600; }}
.wg-pill.hi {{ background:#FBE9E7; color:{C['alert']}; }}
.wg-pill.lo {{ background:#E8F3EE; color:{C['ok']}; }}
footer {{ visibility:hidden; }}
</style>
""")


def cards(items):
    """items: list of (label, value, note, variant)."""
    html = "".join(
        f'<div class="wg-card {v}"><div class="k">{k}</div><div class="v">{val}</div><div class="n">{n}</div></div>'
        for k, val, n, v in items
    )
    st.html(f'<div class="wg-grid">{html}</div>')


def hero(title, sub):
    st.html(f"""<div class="wg-hero"><div><div class="wg-title">{title}</div><div class="wg-sub">{sub}</div></div>
    <span class="wg-status"><span class="wg-dot"></span>Retrospective replay · BattLeDIM 2018 benchmark · not live</span></div>""")


def note(text, kind="note"):
    st.html(f'<div class="wg-{kind}">{text}</div>')


def section(text):
    st.html(f'<div class="wg-sec">{text}</div>')


def style(fig, height=380, legend=True):
    fig.update_layout(
        height=height, template="plotly_white", margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="IBM Plex Sans, sans-serif", size=12, color=C["ink"]),
        hovermode="x unified" if legend else "closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0) if legend else None,
        showlegend=legend, plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#EEF1F5", linecolor=C["line"])
    fig.update_yaxes(gridcolor="#EEF1F5", linecolor=C["line"])
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def shade_severe(fig, df, **kw):
    for _, e in episodes(df["Timestamp"], df["target"]).iterrows():
        fig.add_vrect(x0=e.start, x1=e.end + pd.Timedelta("5min"), fillcolor=C["alert_bg"], line_width=0, layer="below", **kw)


# ------------------------------------------------------------------ data
missing = missing_outputs()
if missing:
    st.error("Missing pre-computed outputs: " + ", ".join(missing) +
             ". Run `python scripts/build_dashboard_data.py` and `python scripts/run_experiments.py` (see README).")
    st.stop()


@st.cache_data
def data():
    return load_all()


D = data()
pred, ctx, imp = D["pred"], D["ctx"], D["imp"]
M = point_metrics(pred["target"], pred["prediction"], pred["risk_probability"])
TEST_START, TEST_END = pred["Timestamp"].min(), pred["Timestamp"].max()


def fmt_ts(t):
    return pd.Timestamp(t).strftime("%d %b %Y %H:%M")


# ================================================================== PAGES
def page_overview():
    hero("WaterGuard AI Saudi",
         "Machine-learning decision support for spotting network conditions associated with elevated water loss, "
         "from hydraulic SCADA signals. Evaluated on the international BattLeDIM L-Town benchmark — not Saudi utility data.")

    cards([
        ("ROC-AUC", f"{M['roc_auc']:.3f}", "ranking quality on unseen test period", "accent"),
        ("Precision", f"{M['precision']:.1%}", "share of alerts that were truly severe", "good"),
        ("Recall", f"{M['recall']:.1%}", "share of severe 5-min steps alerted", "warn"),
        ("Alert threshold", f"{V2_ALERT_THRESHOLD:.2f}", "chosen on validation data only", "accent"),
    ])
    cards([
        ("Test observations", f"{len(pred):,}", f"5-min steps · {TEST_START:%d %b} – {TEST_END:%d %b %Y}", ""),
        ("Severe observations", f"{int(pred['target'].sum()):,}", f"total leakage ≥ {SEVERE_THRESHOLD:.0f} (experimental)", ""),
        ("Detected severe", f"{M['tp']:,}", "true positives", ""),
        ("False alerts", f"{M['fp']:,}", f"of {M['tp'] + M['fp']:,} alerts raised", ""),
    ])

    section("AI risk timeline — test period")
    fig = go.Figure()
    shade_severe(fig, pred)
    fig.add_trace(go.Scattergl(x=pred["Timestamp"], y=pred["risk_probability"], mode="lines", name="Model risk probability",
                               line=dict(color=C["blue"], width=1)))
    fig.add_hline(y=V2_ALERT_THRESHOLD, line_dash="dash", line_color=C["alert"],
                  annotation_text=f"alert threshold {V2_ALERT_THRESHOLD:.2f}", annotation_position="top left")
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=C["alert_bg"], symbol="square"),
                             name="Actual severe condition (ground truth, evaluation only)"))
    fig.update_yaxes(title="Risk probability", range=[0, 1])
    show(style(fig, 420))
    note("Blue: the model's output, computed from SCADA features only. Red bands: benchmark ground truth, shown "
         "<b>only to evaluate</b> the model; it is never a model input. Higher probability means current hydraulic "
         "conditions resemble severe water-loss periods the model saw in training.")

    section("Severe-loss episodes in the test period")
    ev = D["events"].copy()
    ev_disp = pd.DataFrame({
        "Episode start": ev["start"].dt.strftime("%d %b %Y %H:%M"), "Episode end": ev["end"].dt.strftime("%d %b %Y %H:%M"),
        "Duration (days)": (ev["duration_hours"] / 24).round(1), "Alert raised?": np.where(ev["detected"], "Yes", "No"),
        "Delay to first alert (h)": ev["hours_to_first_alert"].round(2),
        "Share of episode alerted": (ev["alerted_share"] * 100).round(1), "Peak risk": ev["max_risk"].round(2),
    })
    st.dataframe(ev_disp, width="stretch", hide_index=True, column_config={
        "Share of episode alerted": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})
    note("The ~9,000 severe 5-minute observations belong to just <b>two</b> long leak episodes. The model raised an alert "
         "within the first 5-minute step of both, but stayed above threshold for only part of each episode — which is why "
         "step-level recall is 38%. Two episodes are too few to generalise event-level performance.")


def page_monitor():
    hero("Risk Monitor", "Filter the retrospective test period, inspect the highest-risk observations and export alerts.")
    c1, c2 = st.columns([2, 1])
    with c1:
        rng = st.date_input("Date range", value=(TEST_START.date(), TEST_END.date()),
                            min_value=TEST_START.date(), max_value=TEST_END.date())
    with c2:
        thr = st.slider("What-if alert threshold", 0.05, 0.80, V2_ALERT_THRESHOLD, 0.01,
                        help="Verified results use 0.22 (chosen on validation data). Changing it here is exploratory: "
                             "choosing a threshold by looking at test data would be a form of data leakage.")
    if not isinstance(rng, (tuple, list)) or len(rng) != 2:
        st.info("Select a start and end date.")
        return
    lo, hi = pd.Timestamp(rng[0]), pd.Timestamp(rng[1]) + pd.Timedelta("1D")
    w = pred[(pred["Timestamp"] >= lo) & (pred["Timestamp"] < hi)].copy()
    if w.empty:
        st.warning("No observations in this range.")
        return
    w["alert"] = (w["risk_probability"] >= thr).astype(int)
    m = point_metrics(w["target"], w["alert"])
    if abs(thr - V2_ALERT_THRESHOLD) > 1e-9:
        note(f"Exploratory threshold {thr:.2f} — verified results use {V2_ALERT_THRESHOLD:.2f}.", "gt")
    cards([
        ("Observations", f"{len(w):,}", f"{w['Timestamp'].min():%d %b} – {w['Timestamp'].max():%d %b %Y}", ""),
        ("AI alerts", f"{int(w['alert'].sum()):,}", f"{w['alert'].mean():.1%} of window", "warn"),
        ("Peak risk", f"{w['risk_probability'].max():.2f}", fmt_ts(w.loc[w['risk_probability'].idxmax(), 'Timestamp']), "accent"),
        ("Alerts confirmed severe", f"{m['tp']:,}" if m['tp'] + m['fp'] else "—",
         f"precision {m['precision']:.1%}" if m['tp'] + m['fp'] else "no alerts", "good"),
    ])

    section("Risk in selected window")
    fig = go.Figure()
    shade_severe(fig, w)
    fig.add_trace(go.Scattergl(x=w["Timestamp"], y=w["risk_probability"], mode="lines", name="Risk probability",
                               line=dict(color=C["blue"], width=1)))
    fig.add_hline(y=thr, line_dash="dash", line_color=C["alert"])
    fig.update_yaxes(title="Risk probability", range=[0, 1])
    show(style(fig, 300, legend=False))

    section("Highest-risk observations")
    top = w.nlargest(100, "risk_probability")
    table = pd.DataFrame({
        "Timestamp": top["Timestamp"].dt.strftime("%Y-%m-%d %H:%M"),
        "Risk probability": top["risk_probability"].round(3),
        "Model status": np.where(top["alert"] == 1, "ALERT", "Normal"),
        "Ground truth: total leakage": top["total_leak"].round(2),
        "Ground truth: severe?": np.where(top["target"] == 1, "Yes", "No"),
    })
    st.dataframe(table, width="stretch", hide_index=True, height=360, column_config={
        "Risk probability": st.column_config.ProgressColumn(format="%.3f", min_value=0, max_value=1)})
    note("Columns marked <b>Ground truth</b> come from the benchmark's leakage file and are shown for retrospective "
         "evaluation only. They were never supplied to the model.", "gt")

    export = w[w["alert"] == 1][["Timestamp", "risk_probability", "total_leak", "target"]].rename(
        columns={"total_leak": "ground_truth_total_leak", "target": "ground_truth_severe"})
    st.download_button("Download alerts in window (CSV)", export.to_csv(index=False).encode(),
                       file_name=f"waterguard_alerts_{rng[0]}_{rng[1]}_thr{thr:.2f}.csv", mime="text/csv",
                       icon=":material/download:", disabled=export.empty)


def page_explain():
    hero("Alert Explainer", "Why did the model flag this moment? A descriptive look at one observation's signals.")
    mode = st.segmented_control("Choose observation", ["Highest-risk alerts", "Any timestamp"], default="Highest-risk alerts")
    if mode == "Any timestamp":
        c1, c2 = st.columns(2)
        d = c1.date_input("Date", value=pd.Timestamp("2018-10-10").date(), min_value=TEST_START.date(), max_value=TEST_END.date())
        times = pd.date_range("00:00", "23:55", freq="5min").strftime("%H:%M").tolist()
        t = c2.selectbox("Time", times, index=times.index("12:00"))
        ts = pd.Timestamp(f"{d} {t}")
    else:
        opts = pred.nlargest(50, "risk_probability")["Timestamp"].tolist()
        ts = st.selectbox("Alert", opts, format_func=lambda x: f"{fmt_ts(x)} — risk {pred.loc[pred.Timestamp == x, 'risk_probability'].iloc[0]:.2f}")
    row = ctx[ctx["Timestamp"] == ts]
    prow = pred[pred["Timestamp"] == ts]
    if row.empty or prow.empty:
        st.warning("Timestamp outside the test period.")
        return
    row, prow = row.iloc[0], prow.iloc[0]
    alert = prow["risk_probability"] >= V2_ALERT_THRESHOLD
    cards([
        ("Observation", fmt_ts(ts), "5-minute SCADA snapshot", ""),
        ("Risk probability", f"{prow['risk_probability']:.3f}", f"threshold {V2_ALERT_THRESHOLD:.2f}", "warn" if alert else "good"),
        ("Model status", "ALERT" if alert else "Normal", "model output", "warn" if alert else "good"),
        ("Ground truth", "Severe" if prow["target"] else "Not severe", f"total leakage {prow['total_leak']:.1f} · evaluation only", ""),
    ])

    typ = D["typical"].set_index("feature")
    sens_cols = [c for c in ctx.columns if c.startswith("sens_")]
    feats = [c.removeprefix("sens_") for c in sens_cols]
    gl_rank = {f: i + 1 for i, f in enumerate(imp["feature"])}
    tab = pd.DataFrame({
        "Signal": feats,
        "Category": [feature_category(f) for f in feats],
        "Current value": [row[f] for f in feats],
        "Typical (median)": [typ.loc[f, "train_normal_median"] for f in feats],
        "Typical range (5–95%)": [f"{typ.loc[f, 'train_normal_q05']:.2f} – {typ.loc[f, 'train_normal_q95']:.2f}" for f in feats],
        "Deviation (σ)": [(row[f] - typ.loc[f, "train_normal_median"]) / typ.loc[f, "train_normal_std"] if typ.loc[f, "train_normal_std"] else 0 for f in feats],
        "Risk change if typical": [row[c] for c in sens_cols],
        "Global rank": [gl_rank.get(f) for f in feats],
    }).sort_values("Risk change if typical", ascending=False)
    tab["Outside typical range"] = [
        "Yes" if (row[f] < typ.loc[f, "train_normal_q05"] or row[f] > typ.loc[f, "train_normal_q95"]) else "No" for f in tab["Signal"]]

    left, right = st.columns([1.1, 1])
    with left:
        section("Local sensitivity — this observation")
        s = tab.sort_values("Risk change if typical")
        fig = go.Figure(go.Bar(x=s["Risk change if typical"], y=s["Signal"], orientation="h",
                               marker_color=[C["alert"] if v > 0 else C["teal"] for v in s["Risk change if typical"]],
                               hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
        fig.update_xaxes(title="Risk probability contributed vs. typical value", zeroline=True, zerolinecolor="#9AA7B6")
        show(style(fig, 420, legend=False))
    with right:
        section("How to read this")
        st.markdown(
            f"- For each of the model's 12 globally most important signals, we replace **only that signal** with its "
            f"typical value (median of *non-severe training* periods) and re-score.\n"
            f"- **Red bars**: the current value of that signal pushes risk *up* relative to typical; **teal**: pushes it down.\n"
            f"- Signals interact inside a Random Forest, so bars need not add up to the total.\n"
            f"- This is a **descriptive, per-observation** view of the model's behaviour. It is different from the "
            f"**global** feature importance on the Sensor Intelligence page, and it does **not** show what caused any leak.")
    section("Signal values vs typical conditions")
    st.dataframe(tab, width="stretch", hide_index=True, column_config={
        "Current value": st.column_config.NumberColumn(format="%.3f"),
        "Typical (median)": st.column_config.NumberColumn(format="%.3f"),
        "Deviation (σ)": st.column_config.NumberColumn(format="%+.2f"),
        "Risk change if typical": st.column_config.NumberColumn(format="%+.3f"),
    })


def page_timeline():
    hero("Detection Timeline", "Benchmark leakage (ground truth) against the model's alerts and risk, on one time axis.")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.55, 0.45],
                        subplot_titles=("Ground truth: total benchmark leakage (evaluation only)", "Model output: risk probability"))
    fig.add_trace(go.Scattergl(x=pred["Timestamp"], y=pred["total_leak"], mode="lines", name="Total leakage (ground truth)",
                               line=dict(color="#475569", width=1)), 1, 1)
    al = pred[pred["prediction"] == 1]
    fig.add_trace(go.Scattergl(x=al["Timestamp"], y=al["total_leak"], mode="markers", name="AI alert",
                               marker=dict(color=C["alert"], size=3)), 1, 1)
    fig.add_hline(y=SEVERE_THRESHOLD, line_dash="dash", line_color=C["amber"], row=1, col=1,
                  annotation_text=f"experimental severity threshold = {SEVERE_THRESHOLD:.0f}", annotation_position="top left")
    fig.add_trace(go.Scattergl(x=pred["Timestamp"], y=pred["risk_probability"], mode="lines", name="Risk probability",
                               line=dict(color=C["blue"], width=1)), 2, 1)
    fig.add_hline(y=V2_ALERT_THRESHOLD, line_dash="dash", line_color=C["alert"], row=2, col=1,
                  annotation_text=f"alert threshold {V2_ALERT_THRESHOLD:.2f}", annotation_position="top left")
    fig.update_yaxes(title="Leakage", row=1, col=1)
    fig.update_yaxes(title="Probability", range=[0, 1], row=2, col=1)
    style(fig, 640)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.08, xanchor="left", x=0), margin=dict(t=40))
    show(fig)
    st.caption("Drag across either panel to zoom; double-click to reset.")
    c1, c2, c3 = st.columns(3)
    with c1:
        note("<b>Top panel</b> is the benchmark's recorded leakage. Values above the amber line count as "
             "\"severe\" in this experiment. Red dots mark times the model raised an alert.")
    with c2:
        note("<b>Bottom panel</b> is what the model produced from SCADA data alone. An alert is raised when the blue "
             "line crosses the red dashed threshold.")
    with c3:
        note("Background leakage of roughly 18–25 units is present almost all year (persistent small leaks), which is why "
             "\"any leakage > 0\" was not a useful target.", "gt")


def page_sensors():
    hero("Sensor Intelligence", "Which hydraulic signals the Random Forest relied on, and how they behave over time.")
    im = imp.copy()
    im["category"] = im["feature"].map(feature_category)
    c1, c2 = st.columns([1.35, 1])
    with c1:
        section("Top 15 features (global Random Forest importance)")
        t = im.head(15).iloc[::-1]
        fig = go.Figure(go.Bar(x=t["importance"], y=t["feature"], orientation="h",
                               marker_color=[CAT_COLORS[c] for c in t["category"]], customdata=t["category"],
                               hovertemplate="%{y}<br>%{customdata}<br>importance %{x:.3f}<extra></extra>"))
        fig.update_xaxes(title="Mean decrease in impurity")
        show(style(fig, 470, legend=False))
    with c2:
        section("Importance by hydraulic category")
        g = im.groupby("category")["importance"].sum().sort_values()
        fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h", marker_color=[CAT_COLORS[c] for c in g.index],
                               text=[f"{v:.0%}" for v in g.values], textposition="outside",
                               hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        fig.update_xaxes(range=[0, g.max() * 1.25], tickformat=".0%")
        show(style(fig, 470, legend=False))
    note("These features contributed strongly to the Random Forest's predictions. Importance measures how much the model "
         "used a signal to split data — it does <b>not</b> prove that a sensor or location caused leakage. Sensor IDs "
         "(e.g. n215) are L-Town benchmark node names, not real Saudi locations.")

    section("Signal explorer")
    typ = D["typical"].set_index("feature")
    options = [c for c in typ.index]
    f = st.selectbox("Signal", options, format_func=lambda x: f"{x} — {feature_category(x)}")
    merged = ctx[["Timestamp", f]].merge(pred[["Timestamp", "target"]], on="Timestamp")
    daily = merged.set_index("Timestamp")[f].resample("1h").mean().reset_index()
    fig = go.Figure()
    shade_severe(fig, merged)
    fig.add_hrect(y0=typ.loc[f, "train_normal_q05"], y1=typ.loc[f, "train_normal_q95"], fillcolor="rgba(14,138,131,0.08)",
                  line_width=0, layer="below")
    fig.add_trace(go.Scattergl(x=daily["Timestamp"], y=daily[f], mode="lines", name=f"{f} (hourly mean)",
                               line=dict(color=C["blue"], width=1.2)))
    fig.add_hline(y=typ.loc[f, "train_normal_median"], line_dash="dot", line_color=C["teal"],
                  annotation_text="typical (training, non-severe median)", annotation_position="top left")
    show(style(fig, 340, legend=False))
    st.caption("Teal band: 5th–95th percentile during non-severe training periods. Red bands: ground-truth severe periods (evaluation only).")

    section("What the signals mean")
    a, b, c = st.columns(3)
    a.markdown("**Pressure (m)** — 33 sensors. A leak lets water escape, so nearby pressure tends to drop and the "
               "spread between sensors (pressure_std, pressure_range) can change.")
    b.markdown("**Flow (m³/h)** — inlet flows (p227, p235) and a pump. Extra water leaving the network through leaks "
               "must be supplied, which can raise inflow, especially at night when demand is low.")
    c.markdown("**Tank level (m)** — tank T1 buffers supply and demand. Unusual filling or draining can reflect extra "
               "outflow. **Demand (L/h)** — 82 metered consumers; used only as network totals.")


def page_performance():
    hero("Model Performance", "Verified results of Random Forest V2 on an untouched, later test period.")
    cards([
        ("ROC-AUC", f"{M['roc_auc']:.4f}", "0.5 = random ranking", "accent"),
        ("Precision", f"{M['precision']:.2%}", "few false alerts", "good"),
        ("Recall", f"{M['recall']:.2%}", "misses many severe steps", "warn"),
        ("F1 score", f"{M['f1']:.4f}", "balance of the two", "accent"),
        ("PR-AUC", f"{M['pr_auc']:.3f}", f"baseline = prevalence {pred['target'].mean():.3f}", ""),
    ])
    assert (M["tn"], M["fp"], M["fn"], M["tp"]) == (V2_VERIFIED["tn"], V2_VERIFIED["fp"], V2_VERIFIED["fn"], V2_VERIFIED["tp"])

    c1, c2 = st.columns([1, 1.2])
    with c1:
        section("Confusion matrix (test, threshold 0.22)")
        z = [[M["tn"], M["fp"]], [M["fn"], M["tp"]]]
        labels = [[f"TN<br><b>{M['tn']:,}</b><br>correctly quiet", f"FP<br><b>{M['fp']:,}</b><br>false alerts"],
                  [f"FN<br><b>{M['fn']:,}</b><br>missed severe", f"TP<br><b>{M['tp']:,}</b><br>detected severe"]]
        fig = go.Figure(go.Heatmap(z=z, x=["Predicted: not severe", "Predicted: severe"], y=["Actual: not severe", "Actual: severe"],
                                   text=labels, texttemplate="%{text}", colorscale=[[0, "#F2F6FB"], [1, "#6FA3D6"]],
                                   showscale=False, hoverinfo="skip"))
        fig.update_yaxes(autorange="reversed")
        show(style(fig, 360, legend=False))
    with c2:
        section("Honest interpretation")
        st.markdown(
            f"- **High precision (94%)**: when V2 raises an alert, it is usually during a severe period — only "
            f"{M['fp']} of {M['tp'] + M['fp']:,} alerts were false.\n"
            f"- **Limited recall (38%)**: {M['fn']:,} of {M['tp'] + M['fn']:,} severe 5-minute steps were missed. The model "
            f"tends to alert at the start of an episode and then lose confidence as conditions persist.\n"
            f"- **Strong ranking (ROC-AUC 0.95)**: severe periods generally receive higher risk than normal ones.\n"
            f"- **Context**: an \"always alert\" rule scores F1 0.44 here, so F1 alone flatters little; precision and "
            f"ranking are where V2 adds value.\n"
            f"- Test prevalence (28.5%) differs from validation (13.0%), so the validation-chosen threshold behaves "
            f"differently on test — validation precision was only 31%.")

    c1, c2, c3 = st.columns(3)
    cur = D["curves"]
    with c1:
        section("ROC curve")
        r = cur[cur.curve == "roc"]
        fig = go.Figure([go.Scatter(x=r.x, y=r.y, mode="lines", line=dict(color=C["blue"]), name="V2"),
                         go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#B8C2CE", dash="dot"), name="random")])
        fig.update_xaxes(title="False positive rate"); fig.update_yaxes(title="True positive rate")
        show(style(fig, 300, legend=False))
    with c2:
        section("Precision–recall curve")
        r = cur[cur.curve == "pr"]
        fig = go.Figure([go.Scatter(x=r.x, y=r.y, mode="lines", line=dict(color=C["teal"])),
                         go.Scatter(x=[M["recall"]], y=[M["precision"]], mode="markers", marker=dict(color=C["alert"], size=10))])
        fig.add_hline(y=pred["target"].mean(), line_dash="dot", line_color="#B8C2CE")
        fig.update_xaxes(title="Recall"); fig.update_yaxes(title="Precision", range=[0, 1.02])
        show(style(fig, 300, legend=False))
    with c3:
        section("Threshold selection (validation only)")
        tv = D["thr_val"]
        fig = go.Figure()
        for col, colr in [("precision", C["ok"]), ("recall", C["alert"]), ("f1", C["blue"])]:
            fig.add_trace(go.Scatter(x=tv.threshold, y=tv[col], mode="lines", name=col.upper() if col == "f1" else col.title(),
                                     line=dict(color=colr)))
        fig.add_vline(x=V2_ALERT_THRESHOLD, line_dash="dash", line_color="#475569")
        fig.update_xaxes(title="Threshold"); fig.update_yaxes(range=[0, 1])
        show(style(fig, 300))

    section("Chronological evaluation design")
    sp = D["audit"]["splits"]
    fig = go.Figure()
    for name, colr in [("train", C["blue"]), ("validation", C["amber"]), ("test", C["teal"])]:
        s = sp[name]
        fig.add_trace(go.Bar(y=["2018"], x=[(pd.Timestamp(s["end"]) - pd.Timestamp(s["start"])).total_seconds() * 1000 + 3e5],
                             base=[pd.Timestamp(s["start"])], orientation="h", marker_color=colr, name=name.title(),
                             text=f"{name.title()} · {s['rows']:,} rows · severe {s['severe_rate']:.1%}", textposition="inside",
                             hovertemplate=f"{s['start']} → {s['end']}<extra></extra>"))
    fig.update_xaxes(type="date"); fig.update_layout(barmode="stack")
    show(style(fig, 160))
    st.caption("No shuffling: the model learns from the past, the alert threshold is tuned on the next period, and the "
               "final score comes from a later period never used for any choice.")

    section("Comparison with baselines (same split, thresholds tuned on validation)")
    b = D["baselines"].copy()
    b = b.rename(columns={"model": "Model", "precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC",
                          "pr_auc": "PR-AUC", "episodes_detected": "Episodes alerted"})
    st.dataframe(b[["Model", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Episodes alerted"]].round(4), width="stretch", hide_index=True)
    note("Logistic Regression ranks test periods <b>worse than random</b> (ROC-AUC 0.22): linear relationships learned from "
         "the training leaks reverse for the test-period leaks, a sign of non-stationary leak signatures. The Random Forest "
         "copes better, but this is a warning that performance may not transfer to different leaks or networks.")

    c1, c2 = st.columns(2)
    with c1:
        section("V1 baseline → V2")
        v1 = D["pred_v1"]
        m1 = point_metrics(v1["severe_leak"], v1["predicted_severe"], v1["risk_probability"])
        st.dataframe(pd.DataFrame({
            "": ["Features", "Month feature", "Split", "Threshold", "Precision", "Recall", "F1", "ROC-AUC"],
            "V1": ["20 aggregates", "Yes (top feature)", "70/30", "default 0.50", f"{m1['precision']:.4f}", f"{m1['recall']:.4f}", f"{m1['f1']:.4f}", f"{m1['roc_auc']:.4f}"],
            "V2": ["60 hydraulic", "Removed", "60/10/30", "0.22 (validation)", f"{M['precision']:.4f}", f"{M['recall']:.4f}", f"{M['f1']:.4f}", f"{M['roc_auc']:.4f}"],
        }), width="stretch", hide_index=True)
    with c2:
        section("Sensitivity to the severity definition")
        s = D["severity"].dropna(subset=["precision"]).sort_values("severity_threshold").copy()
        s = s[["severity_threshold", "test_rate", "precision", "recall", "f1", "roc_auc"]].rename(columns={
            "severity_threshold": "Severe if total leakage ≥", "test_rate": "Test prevalence", "precision": "Precision",
            "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC"})
        st.dataframe(s.round(4), width="stretch", hide_index=True)
        st.caption("Supplementary retraining with other experimental thresholds (37.56 = 80th percentile of training-period "
                   "leakage only). V2 at 40 remains the reported model. At ≥ 50 there are no test positives.")


def page_method():
    hero("Methodology & About", "Research question, data provenance, method, limitations and responsible use.")
    a, b = st.columns([1.2, 1])
    with a:
        st.markdown(f"""
#### Research question
*Can machine-learning analysis of pressure, flow, demand and tank-level patterns identify water-network conditions
associated with elevated water loss — and how reliably, on a later, unseen period?*

#### Why it matters (Saudi context)
Saudi Arabia's Ministry of Environment, Water and Agriculture lists reducing losses in distribution networks as an
improvement opportunity in its National Water Strategy ([MEWA](https://www.mewa.gov.sa/en/Ministry/Agencies/TheWaterAgency/Topics/Pages/Strategy.aspx)).
WaterGuard explores how sensor analytics could support that kind of work. **This motivation is Saudi; the evaluation data is not.**

#### Data provenance
- **BattLeDIM 2020** (Battle of the Leakage Detection and Isolation Methods), L-Town benchmark network, 2018 SCADA
  and leakage files. Vrachimis et al., Zenodo, [doi:10.5281/zenodo.4017659](https://doi.org/10.5281/zenodo.4017659), CC BY 4.0.
- 105,120 timestamps at 5-minute resolution; 33 pressure, 82 demand, 3 flow and 1 tank-level signal; 14 leak locations.
- **Not Saudi utility data.** Sensor names (e.g. n215, p235) are benchmark identifiers, not real locations.

#### Target definition
97.8% of timestamps contain some positive leakage because several small leaks persist all year, so "any leak > 0"
is almost always true. WaterGuard instead flags **total leakage ≥ {SEVERE_THRESHOLD:.0f}** — an *experimental benchmark
severity threshold* (≈ 80th percentile of 2018 total leakage), not an engineering, utility, regulatory or Saudi standard.
""")
    with b:
        st.markdown("""
#### Pipeline
1. Load & validate SCADA (no duplicates, gaps or missing values; all sheets row-aligned)
2. Engineer 60 hydraulic features (current & past values only)
3. Create the target from ground truth — used for training labels and evaluation only
4. Chronological split: train 60% · validation 10% · test 30%
5. Median imputer fitted on training data only
6. Random Forest (300 trees, depth 16, balanced class weights)
7. Alert threshold chosen by best F1 on **validation**
8. One final evaluation on the untouched **test** period
9. Pre-computed outputs → this dashboard

#### Limitations
- Simulated benchmark network; one year; only 2 severe episodes in test.
- Recall is limited (38%); many severe periods are missed.
- The 40 threshold was chosen from the full-year distribution (label definition only; no feature leakage).
- Leak signatures shift between periods (see Logistic Regression result).
- No localisation: the model says *when*, not *where*.

#### Future work
Evaluate on 2019 BattLeDIM data; event-based alert logic (persistence / hysteresis); leak localisation using
the network model; probability calibration; testing with real utility data under a data-sharing agreement.
""")
    note("<b>Responsible use.</b> WaterGuard AI Saudi is an educational research prototype. It is not an operational "
         "utility system, has no partnership with or endorsement from any Saudi utility or government body, and has not "
         "been validated on real Saudi network data. It must not be used for operational decisions.", "gt")


pages = [
    st.Page(page_overview, title="Overview", icon=":material/dashboard:", default=True),
    st.Page(page_monitor, title="Risk Monitor", icon=":material/monitoring:", url_path="monitor"),
    st.Page(page_explain, title="Alert Explainer", icon=":material/troubleshoot:", url_path="explain"),
    st.Page(page_timeline, title="Detection Timeline", icon=":material/timeline:", url_path="timeline"),
    st.Page(page_sensors, title="Sensor Intelligence", icon=":material/sensors:", url_path="sensors"),
    st.Page(page_performance, title="Model Performance", icon=":material/analytics:", url_path="performance"),
    st.Page(page_method, title="Methodology", icon=":material/menu_book:", url_path="methodology"),
]
_test_page = __import__("os").environ.get("WATERGUARD_TEST_PAGE")  # used only by tests/test_app.py
if _test_page:
    globals()[_test_page]()
else:
    st.navigation(pages, position="top").run()
