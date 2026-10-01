"""Shared Streamlit theme, components and data access. No model logic lives here:
everything model-related is delegated to waterguard.service."""
from __future__ import annotations

from dataclasses import dataclass
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from waterguard import service as svc
from waterguard.config import V2_ALERT_THRESHOLD
from waterguard.dashboard_data import load_all
from waterguard.evaluation import episodes
from waterguard.features import feature_category
from waterguard.wording import WORDING

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
BENCHMARK, USER = "Benchmark demo (2018)", "Your analysis"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');
html, body, .stApp, .stMarkdown p, .stMarkdown li, h1, h2, h3, h4, label, input, textarea {{ font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; }}
.block-container {{ padding-top: 4.2rem; max-width: 1320px; }}
h1, h2, h3, h4 {{ color: {C['ink']}; letter-spacing: -0.01em; }}
.wg-hero {{ display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; flex-wrap:wrap;
           border-bottom:1px solid {C['line']}; padding-bottom:.9rem; margin-bottom:1.1rem; }}
.wg-title {{ font-size:1.65rem; font-weight:700; color:{C['ink']}; line-height:1.15; }}
.wg-sub {{ color:{C['muted']}; font-size:.95rem; margin-top:.25rem; max-width:760px; }}
.wg-status {{ display:inline-flex; align-items:center; gap:.45rem; font-size:.78rem; font-weight:600;
             border-radius:999px; padding:.3rem .75rem; max-width:100%; }}
.wg-status.bench {{ color:{C['blue']}; background:#EAF2FB; border:1px solid #CFE0F3; }}
.wg-status.user {{ color:#0B5F59; background:#E6F4F2; border:1px solid #BFE3DE; }}
.wg-status.neutral {{ color:{C['muted']}; background:{C['bg']}; border:1px solid {C['line']}; }}
.wg-dot {{ width:.5rem; height:.5rem; border-radius:50%; background:currentColor; flex:none; }}
.wg-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:.75rem; margin:.2rem 0 1rem; }}
.wg-card {{ background:#fff; border:1px solid {C['line']}; border-radius:10px; padding:.85rem 1rem; }}
.wg-card .k {{ font-size:.74rem; text-transform:uppercase; letter-spacing:.06em; color:{C['muted']}; font-weight:600; }}
.wg-card .v {{ font-family:'IBM Plex Mono', monospace; font-size:1.45rem; color:{C['ink']}; margin-top:.15rem; overflow-wrap:anywhere; }}
.wg-card .n {{ font-size:.78rem; color:{C['muted']}; margin-top:.1rem; }}
.wg-card.accent {{ border-top:3px solid {C['blue']}; }}
.wg-card.warn {{ border-top:3px solid {C['alert']}; }}
.wg-card.good {{ border-top:3px solid {C['ok']}; }}
.wg-card.caution {{ border-top:3px solid {C['amber']}; }}
.wg-note {{ background:{C['bg']}; border-left:3px solid {C['sky']}; padding:.65rem .9rem; border-radius:4px;
           color:#33475B; font-size:.88rem; margin:.4rem 0 .8rem; }}
.wg-gt {{ background:#FFF8EC; border-left:3px solid {C['amber']}; padding:.6rem .9rem; border-radius:4px;
         color:#5A4410; font-size:.86rem; margin:.4rem 0 .8rem; }}
.wg-err {{ background:#FBECEB; border-left:3px solid {C['alert']}; padding:.6rem .9rem; border-radius:4px;
          color:#6B1F1A; font-size:.86rem; margin:.4rem 0 .8rem; }}
.wg-sec {{ font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; color:{C['muted']};
          font-weight:600; margin:1.2rem 0 .4rem; }}
.wg-issue {{ display:flex; gap:.6rem; align-items:flex-start; padding:.45rem 0; border-bottom:1px solid {C['line']}; font-size:.88rem; }}
.wg-tag {{ font-size:.7rem; font-weight:700; letter-spacing:.05em; border-radius:4px; padding:.1rem .4rem; flex:none; }}
.wg-tag.error {{ background:#FBE9E7; color:{C['alert']}; }}
.wg-tag.warning {{ background:#FFF3DC; color:#8A5A00; }}
.wg-tag.info {{ background:#EAF2FB; color:{C['blue']}; }}
.wg-tag.ok {{ background:#E8F3EE; color:{C['ok']}; }}
footer {{ visibility:hidden; }}
</style>
"""


def inject_css():
    st.html(CSS)


# ------------------------------------------------------------------ components
def cards(items):
    """items: list of (label, value, note, variant). Labels may contain trusted HTML."""
    html = "".join(
        f'<div class="wg-card {v}"><div class="k">{k}</div><div class="v">{escape(str(val))}</div><div class="n">{escape(str(n))}</div></div>'
        for k, val, n, v in items)
    st.html(f'<div class="wg-grid">{html}</div>')


def hero(title, sub, badge: tuple[str, str] | None = None):
    kind, text = badge or ("bench", WORDING["not_live"])
    st.html(f"""<div class="wg-hero"><div><div class="wg-title">{title}</div><div class="wg-sub">{sub}</div></div>
    <span class="wg-status {kind}"><span class="wg-dot"></span>{escape(text)}</span></div>""")


def note(text, kind="note"):
    st.html(f'<div class="wg-{kind}">{text}</div>')


def section(text):
    st.html(f'<div class="wg-sec">{text}</div>')


def issue_list(issues):
    rows = "".join(f'<div class="wg-issue"><span class="wg-tag {i.level}">{i.level.upper()}</span>'
                   f'<span>{escape(i.message)}</span></div>' for i in issues)
    st.html(rows or '<div class="wg-issue"><span class="wg-tag ok">OK</span><span>No issues found.</span></div>')


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


def shade_flag(fig, ts, flag, color=None, **kw):
    for _, e in episodes(ts, flag).iterrows():
        fig.add_vrect(x0=e.start, x1=e.end + pd.Timedelta("5min"), fillcolor=color or C["alert_bg"], line_width=0,
                      layer="below", **kw)


def fmt_ts(t):
    return pd.Timestamp(t).strftime("%d %b %Y %H:%M")


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner=False)
def bench():
    return load_all()


@st.cache_resource(show_spinner=False)
def bundle():
    return svc.load_bundle()


@dataclass
class View:
    """What the Risk/Inspection pages need, regardless of where the data came from."""
    kind: str                     # "benchmark" | "user"
    label: str
    pred: pd.DataFrame            # Timestamp, risk_probability, alert (0/1) [+ total_leak, target if labels]
    has_truth: bool
    deviations: pd.DataFrame      # Timestamp + 33 pressure deviations
    signals: pd.DataFrame         # Timestamp + feature values for the explorer
    flags: pd.DataFrame | None = None
    result: svc.AnalysisResult | None = None

    @property
    def badge(self):
        if self.kind == "benchmark":
            return ("bench", WORDING["not_live"])
        return ("user", f"Your analysis · {self.label} · saved V2 model · not live monitoring")

    def explanation(self, ts) -> tuple[pd.DataFrame, dict] | None:
        """Signals table (same columns in both modes) + extra info."""
        if self.kind == "benchmark":
            D = bench()
            row = D["ctx"][D["ctx"]["Timestamp"] == ts]
            if row.empty:
                return None
            row = row.iloc[0]
            typ = D["typical"].set_index("feature")
            sig = [c.removeprefix("sens_") for c in D["ctx"].columns if c.startswith("sens_")]
            rank = {f: i + 1 for i, f in enumerate(D["imp"]["feature"])}
            tab = pd.DataFrame({
                "signal": sig, "category": [feature_category(f) for f in sig],
                "current_value": [row[f] for f in sig],
                "typical_median": [typ.loc[f, "train_normal_median"] for f in sig],
                "typical_q05": [typ.loc[f, "train_normal_q05"] for f in sig],
                "typical_q95": [typ.loc[f, "train_normal_q95"] for f in sig],
                "deviation_sigma": [(row[f] - typ.loc[f, "train_normal_median"]) / typ.loc[f, "train_normal_std"] for f in sig],
                "risk_change_if_typical": [row["sens_" + f] for f in sig],
                "global_importance_rank": [rank.get(f) for f in sig],
            })
            tab["outside_typical_range"] = (tab["current_value"] < tab["typical_q05"]) | (tab["current_value"] > tab["typical_q95"])
            return tab.sort_values("risk_change_if_typical", ascending=False), {}
        i = self.result.row_index(ts)
        if i is None:
            return None
        tab = svc.explain_alert(self.result.X_imputed[i], self.result.X.iloc[i], bundle())
        return tab, {"reduced_context": bool(self.flags["reduced_context"].iloc[i]),
                     "imputed_values": bool(self.flags["imputed_values"].iloc[i])}


def benchmark_view() -> View:
    D = bench()
    p = D["pred"].rename(columns={"prediction": "alert"})
    sig = D["ctx"][["Timestamp"] + D["typical"]["feature"].tolist()]
    return View("benchmark", "BattLeDIM 2018 test period", p, True, D["pdev"], sig)


def user_view() -> View | None:
    a = st.session_state.get("analysis")
    if not a:
        return None
    res: svc.AnalysisResult = a["result"]
    p = res.results.copy()
    p["alert"] = p["alert"].astype(int)
    has_truth = res.ground_truth is not None
    if has_truth:
        p = p.merge(res.ground_truth, on="Timestamp", how="left")
        p["target"] = (p["total_leak"] >= 40).astype("Int64").where(p["total_leak"].notna())
    typ = bench()["typical"]["feature"].tolist()
    return View("user", a["filename"], p, has_truth, res.deviations, res.X[["Timestamp"] + typ], res.flags, res)


def mode_selector(key: str) -> View | None:
    """Benchmark demo vs uploaded analysis. Returns the active view (None = user mode without analysis)."""
    has_user = st.session_state.get("analysis") is not None
    default = st.session_state.get("mode", USER if has_user else BENCHMARK)
    mode = st.segmented_control("Data source", [BENCHMARK, USER], default=default, key=f"{key}_mode",
                                help="Benchmark demo = verified held-out 2018 evaluation. Your analysis = a file you "
                                     "analysed on the Analyze Data page.")
    mode = mode or default
    st.session_state["mode"] = mode
    if mode == USER:
        return user_view()
    return benchmark_view()


def empty_user_state():
    note("<b>No analysis yet.</b> Upload a compatible file on <b>Analyze Data</b> (or use the sample) and run the "
         "analysis; its results then appear here. Switch to <b>Benchmark demo</b> to explore the verified 2018 evaluation.")
    link_to("analyze", "Go to Analyze Data", ":material/upload_file:")


PAGES: dict = {}  # filled by app.py: key -> st.Page, so pages can link to each other


def link_to(key: str, label: str, icon: str | None = None):
    page = PAGES.get(key)
    if page is None:
        return
    try:
        st.page_link(page, label=label, icon=icon)
    except Exception:  # noqa: BLE001 - e.g. a page rendered on its own in tests, without navigation
        pass


def pick_observation(view: View, key: str):
    """Highest-risk alerts or any timestamp in the active data."""
    p = view.pred
    mode = st.segmented_control("Choose observation", ["Highest-risk alerts", "Any timestamp"],
                                default="Highest-risk alerts", key=f"{key}_obs")
    if mode == "Any timestamp":
        lo, hi = p["Timestamp"].min(), p["Timestamp"].max()
        default_day = pd.Timestamp("2018-10-10").date() if view.kind == "benchmark" else lo.date()
        c1, c2 = st.columns(2)
        d = c1.date_input("Date", value=default_day, min_value=lo.date(), max_value=hi.date(), key=f"{key}_date")
        times = p.loc[p["Timestamp"].dt.date == d, "Timestamp"].dt.strftime("%H:%M").tolist()
        if not times:
            st.warning("No observations on this date.")
            return None
        t = c2.selectbox("Time", times, index=times.index("12:00") if "12:00" in times else 0, key=f"{key}_time")
        return pd.Timestamp(f"{d} {t}")
    risk = p.set_index("Timestamp")["risk_probability"]
    opts = p[p["alert"] == 1].nlargest(50, "risk_probability")["Timestamp"].tolist()
    if not opts:
        note("No alerts in this data. Use <b>Any timestamp</b> to inspect a specific moment.")
        return None
    return st.selectbox("Alert", opts, format_func=lambda x: f"{fmt_ts(x)} — risk {risk[x]:.2f}", key=f"{key}_alert")


def network_figure(values: pd.Series | None = None, highlight: list[str] | None = None, leak_xy=None, height=470):
    """L-Town map: links in grey, pressure sensors coloured by deviation (negative = lower than usual)."""
    D = bench()
    links, sens = D["links"], D["sensors"].set_index("id")
    xs = np.column_stack([links["x0"], links["x1"], np.full(len(links), np.nan)]).ravel()
    ys = np.column_stack([links["y0"], links["y1"], np.full(len(links), np.nan)]).ravel()
    fig = go.Figure(go.Scattergl(x=xs, y=ys, mode="lines", line=dict(color="#C9D2DD", width=1), hoverinfo="skip"))
    if values is not None:
        v = values.reindex(sens.index).astype(float)
        fig.add_trace(go.Scatter(
            x=sens["x"], y=sens["y"], mode="markers", customdata=np.c_[sens.index, v],
            marker=dict(size=9 + np.clip(-v.fillna(0).to_numpy(), 0, 6) * 3, color=v, cmin=-4, cmax=4,
                        line=dict(width=1, color="#fff"), colorscale=[[0, C["alert"]], [0.5, "#E9EDF2"], [1, C["blue"]]],
                        colorbar=dict(title=dict(text="Deviation (σ)", side="right"), thickness=10, len=0.7)),
            hovertemplate="%{customdata[0]}<br>deviation %{customdata[1]:+.2f} σ<extra></extra>"))
    if highlight:
        h = sens.loc[highlight]
        fig.add_trace(go.Scatter(x=h["x"], y=h["y"], mode="text", text=[f"<b>{i}</b>" for i in h.index],
                                 textposition="top center", textfont=dict(color=C["ink"], size=12), hoverinfo="skip"))
    if leak_xy is not None:
        fig.add_trace(go.Scatter(x=[leak_xy[0]], y=[leak_xy[1]], mode="markers+text", text=["true leak (evaluation only)"],
                                 textposition="bottom center", textfont=dict(color=C["amber"]), hoverinfo="skip",
                                 marker=dict(symbol="star", size=20, color=C["amber"], line=dict(color="#5A4410", width=1))))
    style(fig, height, legend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(hovermode="closest")
    return fig


def alert_detail(view: View, ts):
    """The alert 'drawer': status, local sensitivity and signal table for one observation."""
    row = view.pred[view.pred["Timestamp"] == ts]
    ex = view.explanation(ts)
    if row.empty or ex is None:
        st.warning("Timestamp not in the active data.")
        return
    r = row.iloc[0]
    tab, info = ex
    alert = bool(r["alert"])
    truth = ("Severe" if r["target"] == 1 else "Not severe", f"total leakage {r['total_leak']:.1f} m³/h · evaluation only") \
        if view.has_truth and pd.notna(r.get("target")) else ("—", "no labels for this data")
    cards([
        ("Observation", f"{ts:%d %b %Y}", f"{ts:%H:%M} · 5-minute snapshot", ""),
        ("Risk probability", f"{r['risk_probability']:.3f}", f"alert threshold {V2_ALERT_THRESHOLD:.2f}", "warn" if alert else "good"),
        ("Model status", "ALERT" if alert else "No alert", "model output", "warn" if alert else "good"),
        ("Ground truth", truth[0], truth[1], ""),
    ])
    if info.get("reduced_context"):
        note("This observation is in the first 30 minutes of the data (or next to a gap), so its change features "
             "were filled from training medians. Treat it as lower confidence.", "gt")
    if info.get("imputed_values"):
        note("Some sensor readings were missing at this time and were filled with training medians.", "gt")
    left, right = st.columns([1.1, 1])
    with left:
        section("Local sensitivity — this observation")
        s = tab.sort_values("risk_change_if_typical")
        fig = go.Figure(go.Bar(x=s["risk_change_if_typical"], y=s["signal"], orientation="h",
                               marker_color=[C["alert"] if v > 0 else C["teal"] for v in s["risk_change_if_typical"]],
                               hovertemplate="%{y}: %{x:+.3f}<extra></extra>"))
        fig.update_xaxes(title="Risk contributed vs. typical value", zeroline=True, zerolinecolor="#9AA7B6")
        show(style(fig, 440, legend=False))
    with right:
        section("How to read this")
        st.markdown(
            f"- For each of {len(tab)} key signals, **only that signal** is replaced by its typical value (median of "
            f"*non-severe training* periods) and the observation is re-scored with the saved model.\n"
            f"- **Red**: the current value pushes risk *up* relative to typical; **teal**: pushes it down.\n"
            f"- Signals interact in a Random Forest, so bars need not add up to the total.\n"
            f"- {WORDING['importance']}")
    section("Signal values vs typical conditions")
    disp = tab.rename(columns={"signal": "Signal", "category": "Category", "current_value": "Current value",
                               "typical_median": "Typical (median)", "deviation_sigma": "Deviation (σ)",
                               "risk_change_if_typical": "Risk change if typical", "global_importance_rank": "Global rank",
                               "outside_typical_range": "Outside typical range"})
    disp["Outside typical range"] = np.where(disp["Outside typical range"].astype(bool), "Yes", "No")
    st.dataframe(disp[["Signal", "Category", "Current value", "Typical (median)", "Deviation (σ)", "Risk change if typical",
                       "Outside typical range", "Global rank"]], width="stretch", hide_index=True, column_config={
        "Current value": st.column_config.NumberColumn(format="%.3f"),
        "Typical (median)": st.column_config.NumberColumn(format="%.3f"),
        "Deviation (σ)": st.column_config.NumberColumn(format="%+.2f"),
        "Risk change if typical": st.column_config.NumberColumn(format="%+.3f")})


def risk_timeline(view: View, height=380, threshold=V2_ALERT_THRESHOLD, data=None):
    p = view.pred if data is None else data
    fig = go.Figure()
    if view.has_truth and "target" in p:
        shade_flag(fig, p["Timestamp"], p["target"].fillna(0).astype(int))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=C["alert_bg"], symbol="square"),
                                 name="Actual severe condition (ground truth, evaluation only)"))
    fig.add_trace(go.Scattergl(x=p["Timestamp"], y=p["risk_probability"], mode="lines", name="Model risk probability",
                               line=dict(color=C["blue"], width=1)))
    fig.add_hline(y=threshold, line_dash="dash", line_color=C["alert"],
                  annotation_text=f"alert threshold {threshold:.2f}", annotation_position="top left")
    fig.update_yaxes(title="Risk probability", range=[0, 1])
    return style(fig, height)
