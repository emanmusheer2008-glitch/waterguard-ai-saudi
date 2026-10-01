"""4. Inspection & Sensors — where to start looking, and which signals the model relies on."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.common import (C, CAT_COLORS, bench, cards, empty_user_state, hero, mode_selector, network_figure, note,
                       pick_observation, section, shade_flag, show, style)
from waterguard import service as svc
from waterguard.config import V2_ALERT_THRESHOLD
from waterguard.features import feature_category
from waterguard.wording import WORDING


def _inspection(view):
    note(f"<b>{WORDING['inspection_guidance']}</b> Each pressure sensor is compared with its usual value at the same "
         f"time of day (median of non-severe <b>training</b> periods). During a large leak pressure often falls across "
         f"much of the network, so what matters is which sensors fall the <b>most</b>. Sensor IDs are L-Town benchmark "
         f"nodes, not real places.")
    ts = pick_observation(view, f"insp_{view.kind}")
    if ts is None:
        return
    row = view.deviations[view.deviations["Timestamp"] == ts]
    prow = view.pred[view.pred["Timestamp"] == ts]
    if row.empty or prow.empty:
        st.warning("Timestamp not in the active data.")
        return
    g = svc.generate_inspection_guidance(row.iloc[0])
    v = pd.Series(g["all_deviations"])
    prow = prow.iloc[0]
    alert = bool(prow["alert"])
    top = [r["sensor"] for r in g["ranking"][:5]]
    cards([
        ("Observation", f"{ts:%d %b %Y}", f"{ts:%H:%M} · 5-minute snapshot", ""),
        ("Model status", "ALERT" if alert else "No alert", f"risk {prow['risk_probability']:.3f} · threshold {V2_ALERT_THRESHOLD:.2f}",
         "warn" if alert else "good"),
        ('Sensors below <span style="text-transform:none">−2σ</span>', f"{g['sensors_below_minus_2_sigma']} of {g['n_sensors']}",
         "time-of-day adjusted", ""),
        ("Start inspection near", g["ranking"][0]["sensor"], f"deviation {g['ranking'][0]['deviation_sigma']:+.1f} σ", "accent"),
    ])
    if not alert:
        note("The model did not raise an alert at this time. Deviations are still shown, but they are not tied to an alert.", "gt")

    leak_xy = None
    if view.kind == "benchmark":
        chk = bench()["inspect_check"]
        ep = chk[(chk["episode_start"] <= ts) & (chk["episode_end"] >= ts)]
        if st.toggle("Show the benchmark's true leak location (evaluation only)", value=False, disabled=ep.empty,
                     key="truth_toggle", help="Available when the timestamp is inside a severe test episode."):
            if not ep.empty:
                leak_xy = tuple(bench()["links"].set_index("id").loc[ep["leaking_pipe"].iloc[0], ["xm", "ym"]])

    left, right = st.columns([1.6, 1])
    with left:
        section("L-Town network — pressure sensors coloured by deviation")
        show(network_figure(v, highlight=top, leak_xy=leak_xy))
        st.caption("Red = lower than usual for this time of day; blue = higher; larger markers = larger drops. "
                   "Map drawn from the benchmark's EPANET model (L-TOWN.inp, CC BY 4.0).")
    with right:
        section("Suggested inspection order")
        perm = bench()["perm"].sort_values("test_auc_drop_mean", ascending=False)
        prank = {f.removeprefix("P_"): i + 1 for i, f in enumerate(perm["feature"])}
        st.dataframe(pd.DataFrame([{"Rank": r["rank"], "Pressure sensor": r["sensor"], "Deviation (σ)": r["deviation_sigma"],
                                    "Model reliance rank (held-out)": prank.get(r["sensor"])} for r in g["ranking"]]),
                     width="stretch", hide_index=True,
                     column_config={"Deviation (σ)": st.column_config.NumberColumn(format="%+.2f")})
        st.caption("Model reliance rank: position among 60 features in held-out permutation importance (benchmark). "
                   "The guidance and the model are separate views and need not agree.")

    if view.kind == "benchmark":
        section("Does the guidance point near the real leaks? (benchmark check, evaluation only)")
        c = bench()["inspect_check"]
        st.dataframe(pd.DataFrame({
            "Severe episode": c["episode_start"].dt.strftime("%d %b") + " – " + c["episode_end"].dt.strftime("%d %b %Y"),
            "Leaking pipe (ground truth)": c["leaking_pipe"], "Most-depressed sensor": c["top_sensor"],
            "Sensor nearest the leak": c["nearest_sensor"], "Its rank of 33": c["nearest_sensor_rank"],
        }), width="stretch", hide_index=True)
        note(f"Averaged over each severe test episode, the sensor physically closest to the leaking pipe ranked "
             f"<b>{', '.join(f'#{r}' for r in c['nearest_sensor_rank'])}</b> of 33 for pressure drop. Encouraging, but "
             f"two episodes on a simulated network, using map distance, are not evidence that WaterGuard can locate leaks.", "gt")
    else:
        note("No leak locations are known for uploaded data, so the guidance cannot be checked here. The benchmark demo "
             "shows how it compared with true leak locations on the 2018 test episodes.")


def _sensors(view):
    D = bench()
    im = D["imp"].copy()
    im["category"] = im["feature"].map(feature_category)
    note("Importance is a property of the trained model, measured on the benchmark. It is the same whichever data source "
         "is selected; the signal explorer below uses the active data. " + WORDING["importance"])
    c1, c2 = st.columns([1.35, 1])
    with c1:
        section("Top 15 features (training-time impurity importance)")
        t = im.head(15).iloc[::-1]
        fig = go.Figure(go.Bar(x=t["importance"], y=t["feature"], orientation="h",
                               marker_color=[CAT_COLORS[c] for c in t["category"]], customdata=t["category"],
                               hovertemplate="%{y}<br>%{customdata}<br>importance %{x:.3f}<extra></extra>"))
        fig.update_xaxes(title="Mean decrease in impurity")
        show(style(fig, 450, legend=False))
    with c2:
        section("Importance by hydraulic category")
        g = im.groupby("category")["importance"].sum().sort_values()
        fig = go.Figure(go.Bar(x=g.values, y=g.index, orientation="h", marker_color=[CAT_COLORS[c] for c in g.index],
                               text=[f"{v:.0%}" for v in g.values], textposition="outside",
                               hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
        fig.update_xaxes(range=[0, g.max() * 1.25], tickformat=".0%")
        show(style(fig, 450, legend=False))

    section("Training-time importance vs held-out reliance")
    pm = D["perm"].copy()
    pm["test_rank"] = pm["test_auc_drop_mean"].rank(ascending=False).astype(int)
    show_f = list(dict.fromkeys(im["feature"].head(8).tolist() + pm.nlargest(8, "test_auc_drop_mean")["feature"].tolist()))
    s = pm.set_index("feature").loc[show_f].sort_values("test_auc_drop_mean")
    c1, c2 = st.columns([1.35, 1])
    with c1:
        fig = go.Figure([go.Bar(y=s.index, x=s["validation_auc_drop_mean"], orientation="h", name="Validation", marker_color=C["sky"]),
                         go.Bar(y=s.index, x=s["test_auc_drop_mean"], orientation="h", name="Test", marker_color=C["blue"])])
        fig.update_xaxes(title="ROC-AUC drop when the feature is shuffled")
        fig.update_layout(barmode="group")
        show(style(fig, 500))
    with c2:
        n215 = pm.set_index("feature").loc["P_n215"]
        st.markdown(
            f"**Permutation importance** shuffles one feature on data the model never trained on and measures how much "
            f"ranking quality drops: *does the model need this signal on new data?*\n\n"
            f"**P_n215** is #1 by impurity importance but #{int(n215['test_rank'])} of 60 on the test period (AUC drop "
            f"{n215['test_auc_drop_mean']:.3f}). It reads almost constantly about 39.09 m and dropped only during the "
            f"March training leak, so the forest split on it heavily. **P_n229** is the most relied-on signal on both "
            f"validation and test.")
        st.caption("Computed after the fact for description only; nothing about the model or threshold was chosen from it.")

    section("Signal explorer — active data")
    typ = D["typical"].set_index("feature")
    f = st.selectbox("Signal", list(typ.index), format_func=lambda x: f"{x} — {feature_category(x)}", key=f"sig_{view.kind}")
    sig = view.signals[["Timestamp", f]].set_index("Timestamp")[f]
    hourly = sig.resample("1h").mean().reset_index() if len(sig) > 2000 else sig.reset_index()
    fig = go.Figure()
    if view.has_truth:
        shade_flag(fig, view.pred["Timestamp"], view.pred["target"].fillna(0).astype(int))
    fig.add_hrect(y0=typ.loc[f, "train_normal_q05"], y1=typ.loc[f, "train_normal_q95"], fillcolor="rgba(14,138,131,0.08)",
                  line_width=0, layer="below")
    fig.add_trace(go.Scattergl(x=hourly["Timestamp"], y=hourly[f], mode="lines", line=dict(color=C["blue"], width=1.2)))
    fig.add_hline(y=typ.loc[f, "train_normal_median"], line_dash="dot", line_color=C["teal"],
                  annotation_text="typical (training, non-severe median)", annotation_position="top left")
    show(style(fig, 320, legend=False))
    st.caption("Teal band: 5th–95th percentile during non-severe training periods."
               + (" Red bands: ground-truth severe periods (evaluation only)." if view.has_truth else "")
               + (" Hourly means." if len(sig) > 2000 else ""))

    section("What the signals mean")
    a, b, c = st.columns(3)
    a.markdown("**Pressure (m)** — 33 sensors. A leak lets water escape, so nearby pressure tends to drop and the spread "
               "between sensors (pressure_std, pressure_range) can change.")
    b.markdown("**Flow (m³/h)** — inlet flows (p227, p235) and a pump. Water lost through leaks must be supplied, which "
               "can raise inflow, especially at night when demand is low.")
    c.markdown("**Tank level (m)** — tank T1 buffers supply and demand. Unusual filling or draining can reflect extra "
               "outflow. **Demand (L/h)** — 82 meters; used only as network totals.")


def render():
    view = mode_selector("inspect")
    hero("Inspection & Sensors", "Where an operator could start looking, and which hydraulic signals the model relies on.",
         badge=view.badge if view else ("user", "Your analysis · no data yet"))
    if view is None:
        empty_user_state()
        return
    part = st.segmented_control("View", ["Inspection guidance", "Sensor intelligence"], default="Inspection guidance",
                                key=f"insp_view_{view.kind}") or "Inspection guidance"
    _inspection(view) if part == "Inspection guidance" else _sensors(view)
