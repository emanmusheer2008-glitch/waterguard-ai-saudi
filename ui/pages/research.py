"""5. Model & Research — verified performance, methodology, dataset & Saudi context, limitations."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.common import C, bench, cards, hero, note, section, show, style
from waterguard.config import SEVERE_THRESHOLD, V2_ALERT_THRESHOLD, V2_VERIFIED
from waterguard.evaluation import point_metrics
from waterguard.wording import WORDING


def _performance():
    D = bench()
    pred = D["pred"]
    M = point_metrics(pred["target"], pred["prediction"], pred["risk_probability"])
    assert (M["tn"], M["fp"], M["fn"], M["tp"]) == (V2_VERIFIED["tn"], V2_VERIFIED["fp"], V2_VERIFIED["fn"], V2_VERIFIED["tp"])
    cards([
        ("ROC-AUC", f"{M['roc_auc']:.4f}", "0.5 = random ranking", "accent"),
        ("Precision", f"{M['precision']:.2%}", "few false alerts", "good"),
        ("Recall", f"{M['recall']:.2%}", "misses many severe steps", "warn"),
        ("F1 score", f"{M['f1']:.4f}", "balance of the two", "accent"),
        ("PR-AUC", f"{M['pr_auc']:.3f}", f"baseline = prevalence {pred['target'].mean():.3f}", ""),
    ])
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
        show(style(fig, 340, legend=False))
    with c2:
        section("Honest interpretation")
        st.markdown(
            f"- **High precision (94%)**: only {M['fp']} of {M['tp'] + M['fp']:,} alerts were false.\n"
            f"- **Limited recall (38%)**: {M['fn']:,} of {M['tp'] + M['fn']:,} severe 5-minute steps were missed; the model "
            f"tends to alert at an episode's start and then lose confidence.\n"
            f"- **Strong ranking (ROC-AUC 0.95)**: severe periods generally receive higher risk than normal ones.\n"
            f"- **Context**: \"always alert\" scores F1 0.44 here, so F1 alone flatters little.\n"
            f"- Test prevalence (28.5%) differs from validation (13.0%), so the validation-chosen threshold behaves "
            f"differently on test (validation precision was 31%).")
    c1, c2, c3 = st.columns(3)
    cur = D["curves"]
    with c1:
        section("ROC curve")
        r = cur[cur.curve == "roc"]
        fig = go.Figure([go.Scatter(x=r.x, y=r.y, mode="lines", line=dict(color=C["blue"])),
                         go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#B8C2CE", dash="dot"))])
        fig.update_xaxes(title="False positive rate"); fig.update_yaxes(title="True positive rate")
        show(style(fig, 290, legend=False))
    with c2:
        section("Precision–recall curve")
        r = cur[cur.curve == "pr"]
        fig = go.Figure([go.Scatter(x=r.x, y=r.y, mode="lines", line=dict(color=C["teal"])),
                         go.Scatter(x=[M["recall"]], y=[M["precision"]], mode="markers", marker=dict(color=C["alert"], size=10))])
        fig.add_hline(y=pred["target"].mean(), line_dash="dot", line_color="#B8C2CE")
        fig.update_xaxes(title="Recall"); fig.update_yaxes(title="Precision", range=[0, 1.02])
        show(style(fig, 290, legend=False))
    with c3:
        section("Threshold selection (validation only)")
        tv = D["thr_val"]
        fig = go.Figure([go.Scatter(x=tv.threshold, y=tv[c], mode="lines", name=n, line=dict(color=k))
                         for c, n, k in [("precision", "Precision", C["ok"]), ("recall", "Recall", C["alert"]), ("f1", "F1", C["blue"])]])
        fig.add_vline(x=V2_ALERT_THRESHOLD, line_dash="dash", line_color="#475569")
        fig.update_xaxes(title="Threshold"); fig.update_yaxes(range=[0, 1])
        show(style(fig, 290))

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
    st.caption("No shuffling: learn from the past, tune the threshold on the next period, score once on a later period.")

    section("Baselines (same split, thresholds tuned on validation)")
    b = D["baselines"].rename(columns={"model": "Model", "precision": "Precision", "recall": "Recall", "f1": "F1",
                                       "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "episodes_detected": "Episodes alerted"})
    st.dataframe(b[["Model", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Episodes alerted"]].round(4), width="stretch", hide_index=True)
    note("Logistic Regression ranks the test period <b>worse than random</b> (ROC-AUC 0.22): relationships learned from "
         "training-period leaks reverse for test-period leaks, a sign of non-stationary leak signatures.")
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
        s = D["severity"].dropna(subset=["precision"]).sort_values("severity_threshold")
        st.dataframe(s[["severity_threshold", "test_rate", "precision", "recall", "f1", "roc_auc"]].rename(columns={
            "severity_threshold": "Severe if total leakage ≥ (m³/h)", "test_rate": "Test prevalence", "precision": "Precision",
            "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC"}).round(4), width="stretch", hide_index=True)
        st.caption("Supplementary retraining (37.56 = training-only 80th percentile). V2 at 40 remains the reported model.")


def _method():
    st.markdown(f"""
#### Research question
*Can machine-learning analysis of pressure, flow, demand and tank-level patterns identify network conditions
associated with severe water-loss periods in a benchmark distribution network — and how reliably, on a later,
unseen period?*

#### Target
{WORDING['experimental_threshold']} 97.8% of timestamps contain *some* leakage (small leaks persist all year), so
"any leak > 0" was rejected as a target.

#### Pipeline
1. Load & validate SCADA (no duplicates, gaps or missing values in 2018; all sheets row-aligned)
2. Engineer 60 hydraulic features from current & past values only (33 pressures, pressure/demand/flow summaries,
   3 flows, tank level, hour of day as sin/cos, 5- and 30-minute changes; **no month**)
3. Target from ground truth — labels and evaluation only, never a feature
4. Chronological split: train 60% (to 7 Aug) · validation 10% (to 13 Sep) · test 30% (to 31 Dec)
5. Median imputer fitted on training data only
6. Random Forest: 300 trees, depth 16, min leaf 4, `balanced_subsample`, seed 42
7. Alert threshold = best F1 on **validation** → 0.22
8. One evaluation on the untouched **test** period

#### Training vs inference
**Training** (done once, offline: `src/train_v2.py`) needs labelled history and produces the saved model, imputer,
feature list and threshold. **Inference** (Analyze Data, the API) only *applies* those saved artifacts to new SCADA
data: no labels, no retraining, identical feature code. Verified: on the 2018 test period the inference service
reproduces the benchmark probabilities to within 2.2e-16.
""")


def _context():
    c1, c2 = st.columns(2)
    with c1:
        note(f"<b>Saudi context = motivation.</b><br>{WORDING['saudi_motivation']}")
    with c2:
        note(f"<b>BattLeDIM L-Town = experimental data.</b><br>{WORDING['benchmark_disclaimer']}", "gt")
    st.markdown("""
#### Dataset
- **BattLeDIM 2020**, L-Town network, 2018 historical files (Vrachimis et al., 2020, Zenodo,
  [doi:10.5281/zenodo.4017659](https://doi.org/10.5281/zenodo.4017659), CC BY 4.0). Competition site:
  [battledim.ucy.ac.cy](https://battledim.ucy.ac.cy/).
- 105,120 timestamps at 5-minute resolution: 33 pressures (m), 82 demands (L/h), 3 flows (m³/h), 1 tank level (m);
  leakage at 14 pipes (m³/h). The leakage CSV uses `;` separators and decimal commas.
- Sensor names (n215, p235, …) are benchmark identifiers, not real places.

#### Sources
- MEWA, *National Water Strategy* page, last edited 6 Jul 2025 —
  [mewa.gov.sa](https://www.mewa.gov.sa/en/Ministry/Agencies/TheWaterAgency/Topics/Pages/Strategy.aspx).
- Vrachimis, S. G. et al. (2020). *Dataset of BattLeDIM* (v1). Zenodo. CC BY 4.0.
""")


def _other_data():
    st.markdown(f"""
#### 2018 vs 2026 timestamps
{WORDING['time_recency']} Verified with the sample file: L-Town measurements shifted to October 2026 give exactly the
same probabilities as the 2018 benchmark from the 7th row onward (the first 30 minutes lack history for the change
features and are flagged *reduced context*).

#### Time recency is not network compatibility
{WORDING['different_network']}

A different (for example Saudi) network may differ in sensor IDs, topology, pressure distributions, demand patterns,
flow characteristics, tank configuration and operations. The upload validator therefore requires the exact L-Town
sensor schema and also checks whether values fall inside the range seen in training.

#### What a different network would need
""")
    steps = ["Historical SCADA from that network", "Verified incident / leak records (labels)",
             "Network-specific preprocessing and features", "Training or adaptation", "Chronological validation",
             "Threshold selection on validation", "Deployment", "Inference on incoming SCADA"]
    st.html('<div style="display:flex;flex-wrap:wrap;gap:.4rem;align-items:center;margin:.3rem 0 1rem">' + "".join(
        f'<span style="border:1px solid #CFE0F3;background:#F5F9FD;border-radius:6px;padding:.35rem .6rem;font-size:.85rem">'
        f'<b>{i + 1}</b>&nbsp; {s}</span>' + ('<span style="color:#9AA7B6">→</span>' if i < len(steps) - 1 else "")
        for i, s in enumerate(steps)) + "</div>")
    st.markdown("#### Possible future real-world architecture (does not exist today)\n"
                "Utility SCADA stream → real-time validation & features → trained model → risk score → operator "
                "dashboard → field inspection → inspection outcome fed back as new labels.")


def _limits():
    st.markdown("""
- Simulated benchmark network; one year; only **two** severe episodes in the test period (five in the year).
- **Recall is 38%**; many severe periods are missed.
- The 40 m³/h threshold was read from the full-year distribution (label definition only; a training-only percentile,
  37.56, gives similar results).
- Leak signatures shift between periods; the top impurity feature (n215) contributes little on held-out data.
- The model says *when*, not *where*; inspection guidance is a heuristic checked on two episodes.
- Probabilities are not calibrated; 5-minute steps are strongly correlated.
- Only L-Town-compatible data can be analysed.
""")
    note(f"<b>Responsible use.</b> {WORDING['responsible_use']} {WORDING['not_live']}", "gt")


def render():
    hero("Model & Research", "Verified performance, methodology, data provenance, limitations and responsible use.",
         badge=("bench", "Verified benchmark evaluation · BattLeDIM 2018"))
    tabs = st.tabs(["Performance", "Methodology", "Dataset & Saudi context", "Using other data", "Limitations & responsible use"])
    with tabs[0]:
        _performance()
    with tabs[1]:
        _method()
    with tabs[2]:
        _context()
    with tabs[3]:
        _other_data()
    with tabs[4]:
        _limits()
