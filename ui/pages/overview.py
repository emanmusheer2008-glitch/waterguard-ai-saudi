"""1. Overview — what WaterGuard is, the two modes, verified benchmark headline, limitations."""
import numpy as np
import pandas as pd
import streamlit as st

from ui.common import C, bench, cards, hero, link_to, note, risk_timeline, section, show, benchmark_view
from waterguard.config import SEVERE_THRESHOLD, V2_ALERT_THRESHOLD
from waterguard.evaluation import point_metrics
from waterguard.wording import WORDING


def render():
    D = bench()
    pred = D["pred"]
    M = point_metrics(pred["target"], pred["prediction"], pred["risk_probability"])
    hero("WaterGuard AI Saudi",
         "An ML water-loss decision-support research prototype. It reads hydraulic SCADA signals every 5 minutes, "
         "estimates the risk of a severe water-loss period, explains each alert and suggests which pressure sensors to "
         "inspect first.", badge=("neutral", "Research prototype · BattLeDIM L-Town benchmark · not live"))
    note(f"<b>Data disclaimer.</b> {WORDING['benchmark_disclaimer']}", "gt")

    section("Two ways to use WaterGuard")
    a, b = st.columns(2)
    with a:
        with st.container(border=True):
            st.markdown("#### Benchmark demo")
            st.markdown("The verified evaluation: the model was trained on Jan–Aug 2018, its alert threshold tuned on "
                        "Aug–Sep, and it was scored **once** on the untouched Sep–Dec 2018 period. Ground truth is "
                        "available here, so you can see what the model caught and what it missed.")
            link_to("alerts", "Explore benchmark alerts", ":material/monitoring:")
    with b:
        with st.container(border=True):
            st.markdown("#### Analyze data")
            st.markdown("Upload L-Town-compatible SCADA data (CSV or XLSX, 5-minute steps). WaterGuard validates it, "
                        "builds the same 60 features, runs the saved V2 model and threshold — **no retraining, no labels "
                        "needed** — and returns risk, alerts, explanations and an exportable predictions file.")
            link_to("analyze", "Analyze a file", ":material/upload_file:")

    section("Verified benchmark result — held-out test period, 13 Sep – 31 Dec 2018")
    note(f"<b>What is a severe water-loss period?</b> {WORDING['experimental_threshold']}")
    cards([
        ("Precision", f"{M['precision']:.1%}", "share of alerts that were truly severe", "good"),
        ("Recall", f"{M['recall']:.1%}", "share of severe 5-min steps alerted", "warn"),
        ("ROC-AUC", f"{M['roc_auc']:.3f}", "ranking quality on unseen data", "accent"),
        ("Alert threshold", f"{V2_ALERT_THRESHOLD:.2f}", "chosen on validation data only", "accent"),
    ])
    note(f"<b>Read the two key numbers together.</b> {WORDING['precision']} {WORDING['recall']} That is why WaterGuard "
         f"is decision support, not an autonomous leak detector.", "gt")
    cards([
        ("Test observations", f"{len(pred):,}", "5-minute steps", ""),
        ("Severe observations", f"{int(pred['target'].sum()):,}", f"total leakage ≥ {SEVERE_THRESHOLD:.0f} m³/h", ""),
        ("Detected severe", f"{M['tp']:,}", "true positives", ""),
        ("False alerts", f"{M['fp']:,}", f"of {M['tp'] + M['fp']:,} alerts raised", ""),
    ])

    section("Risk timeline — benchmark test period")
    show(risk_timeline(benchmark_view(), 380))
    note("Blue: model output from SCADA features only. Red bands: benchmark ground truth, shown <b>only to evaluate</b> "
         "the model. The ~9,000 severe steps come from just <b>two</b> leak episodes; WaterGuard alerted at the first "
         "5-minute step of both but stayed above threshold for only 48% and 26% of them.")

    ev = D["events"]
    st.dataframe(pd.DataFrame({
        "Severe episode": ev["start"].dt.strftime("%d %b %H:%M") + " – " + ev["end"].dt.strftime("%d %b %Y %H:%M"),
        "Duration (days)": (ev["duration_hours"] / 24).round(1),
        "Alert raised?": np.where(ev["detected"], "Yes", "No"),
        "Delay to first alert (h)": ev["hours_to_first_alert"].round(2),
        "Share of episode alerted": (ev["alerted_share"] * 100).round(1),
        "Peak risk": ev["max_risk"].round(2),
    }), width="stretch", hide_index=True, column_config={
        "Share of episode alerted": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)})

    section("How it works")
    st.markdown("**SCADA data** (33 pressures, 82 demands, 3 flows, 1 tank) → **validation** → **60 hydraulic features** "
                "(current and past values only) → **saved imputer** → **saved Random Forest** → **risk probability** → "
                "**alert if ≥ 0.22** → **explanation** and **inspection guidance**.")

    section("Important limitations")
    l1, l2 = st.columns(2)
    l1.markdown("- Simulated benchmark network, one year of data; only **two** severe episodes in the test period.\n"
                "- **Recall is 38%**: most severe time is missed.\n"
                "- The 40 m³/h severity threshold is experimental.")
    l2.markdown("- The model says *when*, not *where*; inspection guidance is a heuristic.\n"
                "- Works only on data from the **L-Town** network; another network needs retraining.\n"
                "- Probabilities are not calibrated.")
    note(f"<b>Responsible use.</b> {WORDING['responsible_use']}", "gt")
