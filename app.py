from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="WaterGuard AI Saudi",
    page_icon="💧",
    layout="wide"
)

# ------------------------------------------------------------
# PATHS / DATA
# ------------------------------------------------------------
BASE = Path(__file__).resolve().parent
OUTPUTS = BASE / "outputs"

@st.cache_data
def load_data():
    predictions = pd.read_csv(
        OUTPUTS / "predictions_v2.csv",
        parse_dates=["Timestamp"]
    )

    importance = pd.read_csv(
        OUTPUTS / "feature_importance_v2.csv"
    )

    return predictions, importance

df, importance = load_data()

# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------
MODEL_THRESHOLD = 0.22
SEVERE_THRESHOLD = 40.0

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.title("💧 WaterGuard AI")
st.sidebar.caption("Saudi Water-Loss Decision Support Prototype")

page = st.sidebar.radio(
    "Navigation",
    [
        "Network Overview",
        "Leak Risk Monitor",
        "Detection Timeline",
        "Sensor Intelligence",
        "Model Performance",
        "About"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Evaluated using the BattLeDIM L-Town international "
    "water-network leakage benchmark."
)

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.title("💧 WaterGuard AI Saudi")
st.markdown(
    "**AI-powered water-loss risk detection and decision support**"
)

# ------------------------------------------------------------
# NETWORK OVERVIEW
# ------------------------------------------------------------
if page == "Network Overview":

    st.subheader("Network Intelligence Overview")

    st.write(
        "WaterGuard analyzes hydraulic sensor patterns to identify "
        "periods associated with elevated water-loss conditions."
    )

    total = len(df)
    actual = int(df["target"].sum())
    detected = int(
        ((df["target"] == 1) & (df["prediction"] == 1)).sum()
    )
    false_alerts = int(
        ((df["target"] == 0) & (df["prediction"] == 1)).sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Test Observations", f"{total:,}")
    c2.metric("Severe-Loss Periods", f"{actual:,}")
    c3.metric("Detected Severe Periods", f"{detected:,}")
    c4.metric("False Alerts", f"{false_alerts:,}")

    st.divider()

    st.subheader("AI Risk Timeline")

    chart = df[
        ["Timestamp", "risk_probability"]
    ].copy()

    fig = px.line(
        chart,
        x="Timestamp",
        y="risk_probability",
        labels={
            "risk_probability": "AI Risk Probability",
            "Timestamp": "Time"
        }
    )

    fig.add_hline(
        y=MODEL_THRESHOLD,
        line_dash="dash",
        annotation_text="Alert threshold = 0.22"
    )

    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Higher probability indicates that the observed hydraulic "
        "conditions more closely resemble severe water-loss periods "
        "learned by the model."
    )

# ------------------------------------------------------------
# RISK MONITOR
# ------------------------------------------------------------
elif page == "Leak Risk Monitor":

    st.subheader("Leak Risk Monitor")

    risk = df.copy()

    risk["status"] = risk["risk_probability"].apply(
        lambda x:
        "HIGH RISK" if x >= MODEL_THRESHOLD
        else "NORMAL"
    )

    high = risk[risk["status"] == "HIGH RISK"]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "AI Alerts",
        f"{len(high):,}"
    )

    c2.metric(
        "Maximum Risk",
        f"{risk['risk_probability'].max():.1%}"
    )

    c3.metric(
        "Alert Threshold",
        f"{MODEL_THRESHOLD:.0%}"
    )

    st.divider()

    st.subheader("Highest-Risk Observations")

    display = risk.nlargest(
        25, "risk_probability"
    )[
        [
            "Timestamp",
            "risk_probability",
            "total_leak",
            "target",
            "prediction"
        ]
    ].copy()

    display["risk_probability"] = (
        display["risk_probability"] * 100
    ).round(1)

    display = display.rename(
        columns={
            "risk_probability": "AI Risk (%)",
            "total_leak": "Observed Leak Magnitude",
            "target": "Actual Severe Condition",
            "prediction": "AI Alert"
        }
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Leakage measurements are shown here only for retrospective "
        "evaluation. They were not supplied to the model as input features."
    )

# ------------------------------------------------------------
# DETECTION TIMELINE
# ------------------------------------------------------------
elif page == "Detection Timeline":

    st.subheader("Actual vs AI-Detected Water-Loss Conditions")

    timeline = df[
        [
            "Timestamp",
            "total_leak",
            "risk_probability",
            "target",
            "prediction"
        ]
    ].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=timeline["Timestamp"],
            y=timeline["total_leak"],
            name="Observed Total Leakage",
            mode="lines"
        )
    )

    fig.add_hline(
        y=SEVERE_THRESHOLD,
        line_dash="dash",
        annotation_text="V1 severe-loss definition = 40"
    )

    detected = timeline[
        timeline["prediction"] == 1
    ]

    fig.add_trace(
        go.Scatter(
            x=detected["Timestamp"],
            y=detected["total_leak"],
            mode="markers",
            name="AI Alerts"
        )
    )

    fig.update_layout(
        height=520,
        xaxis_title="Time",
        yaxis_title="Observed Leakage Magnitude"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "The leakage magnitude is ground truth used to evaluate the "
        "AI system. The model itself predicts from SCADA-derived features."
    )

# ------------------------------------------------------------
# SENSOR INTELLIGENCE
# ------------------------------------------------------------
elif page == "Sensor Intelligence":

    st.subheader("What Signals Drive WaterGuard?")

    top = importance.head(15).copy()

    fig = px.bar(
        top.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        labels={
            "importance": "Random Forest Importance",
            "feature": "Hydraulic Feature"
        }
    )

    fig.update_layout(height=550)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        """
        **Interpretation**

        WaterGuard V2 relies strongly on pressure-related hydraulic
        signals rather than simply memorizing the month in which
        leakage occurred.

        The most influential signals include individual pressure
        sensors, network pressure variability and flow measurements.

        Feature importance indicates which variables influenced the
        Random Forest most; it does **not** prove that a sensor caused
        a leakage event.
        """
    )

# ------------------------------------------------------------
# MODEL PERFORMANCE
# ------------------------------------------------------------
elif page == "Model Performance":

    st.subheader("Model Evaluation")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("ROC-AUC", "0.950")
    c2.metric("Precision", "94.2%")
    c3.metric("Recall", "38.2%")
    c4.metric("F1 Score", "0.543")

    st.divider()

    st.subheader("Confusion Matrix")

    matrix = pd.DataFrame(
        [
            [22332, 210],
            [5561, 3433]
        ],
        index=[
            "Actual Non-Severe",
            "Actual Severe"
        ],
        columns=[
            "Predicted Non-Severe",
            "Predicted Severe"
        ]
    )

    fig = px.imshow(
        matrix,
        text_auto=True,
        labels=dict(
            x="Prediction",
            y="Actual",
            color="Observations"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown(
        """
        ### Evaluation design

        The model was evaluated chronologically rather than using a
        random train/test split.

        - **Training:** first 60% of 2018
        - **Validation:** following 10%
        - **Testing:** final 30%
        - Alert threshold selected using validation data only
        - Final test period remained untouched during threshold selection

        ### Current limitation

        The model achieves high precision but moderate recall.
        This means its severe-loss alerts are relatively reliable,
        but it still misses a substantial number of severe periods.

        Future work should therefore focus on improving detection
        coverage without creating excessive false alarms.
        """
    )

# ------------------------------------------------------------
# ABOUT
# ------------------------------------------------------------
elif page == "About":

    st.subheader("About WaterGuard AI Saudi")

    st.markdown(
        """
        **WaterGuard AI Saudi** is a machine-learning decision-support
        prototype exploring whether hydraulic sensor patterns can help
        identify elevated water-loss conditions.

        ### Research question

        *Can machine-learning analysis of pressure, flow, demand and
        tank-level patterns identify water-network conditions associated
        with significant leakage?*

        ### Data

        The prototype is evaluated using the **BattLeDIM L-Town**
        international leakage-detection benchmark.

        The dataset is **not Saudi utility data**. WaterGuard is a
        Saudi-focused prototype tested on an international research
        benchmark.

        ### AI methodology

        1. Process SCADA measurements
        2. Engineer hydraulic and temporal features
        3. Train a Random Forest classifier
        4. Tune the alert threshold on a separate validation period
        5. Evaluate on a later, untouched test period
        6. Present risk information through a decision-support interface

        ### Important limitation

        The current severe-loss target uses a data-derived threshold of
        total leakage ≥ 40. This is an experimental definition for the
        benchmark and is **not an official Saudi operational threshold**.

        WaterGuard is a research and educational prototype, not an
        operational utility control system.
        """
    )
