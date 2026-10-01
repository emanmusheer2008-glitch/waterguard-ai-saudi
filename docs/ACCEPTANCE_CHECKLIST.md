# Manual acceptance checklist

Run on your own computer (about 20 minutes):

```bat
cd %USERPROFILE%\OneDrive\Desktop\waterguard-ai-saudi
.venv\Scripts\activate
pip install -r requirements-dev.txt
python -m pytest -q
streamlit run app.py
```

Expected: pytest ends with `passed` and **0 failed**; the browser opens at http://localhost:8501.

| # | Check | Where | Pass if… |
|---|---|---|---|
| 1 | Open Network Overview | first page | KPI cards load: precision 94.2%, recall 38.2%, ROC-AUC 0.950, threshold 0.22 |
| 2 | Understand the dataset | Overview subtitle, Methodology | "BattLeDIM L-Town benchmark — not Saudi utility data" is visible |
| 3 | Saudi vs L-Town distinction | Methodology, top two boxes | Blue box = Saudi motivation (MEWA); amber box = benchmark data |
| 4 | Inspect severe-risk periods | Overview episode table; Risk Monitor table | Two episodes (Oct and Oct–Nov 2018); highest-risk table loads |
| 5 | Change the time window | Risk Monitor → Date range | Pick 6–23 Oct 2018; cards and chart update |
| 6 | Understand predicted risk | Risk Monitor / Alert Explainer | Probability, threshold and ALERT/Normal status are shown |
| 7 | Inspect sensor behaviour | Alert Explainer; Inspection Guidance | Signal table and network map load; the toggle shows a star at the true leak |
| 8 | Detection timeline | Detection Timeline | Leakage (top) with red alert dots; risk (bottom) with 0.22 line; zoom works |
| 9 | Model performance | Model Performance | Confusion matrix TN 22,332 · FP 210 · FN 5,561 · TP 3,433 |
| 10 | Understand precision | Overview amber note | You can say in your own words: "when it alerts, it's right 94% of the time" |
| 11 | Understand recall | Overview amber note | "It caught 38% of severe 5-minute steps; it missed 5,561" |
| 12 | Understand threshold 0.22 | Model Performance → Threshold selection chart | The dashed line is at the validation F1 peak; the what-if slider is labelled exploratory |
| 13 | Understand ≥ 40 | Overview blue note; Methodology | It is called an experimental threshold, not a standard |
| 14 | Limitations | Methodology | Two test episodes, recall 38%, no localisation, uncalibrated, etc. |
| 15 | Not pretending to be live | Every page | The badge reads "Retrospective replay · BattLeDIM 2018 benchmark · not live" |
| 16 | Download works | Risk Monitor → Download alerts | A CSV downloads |
| 17 | Terminal is clean | the window running Streamlit | No red tracebacks or deprecation warnings while clicking through all 8 pages |

If anything fails, copy the error text and bring it back to Claude.
