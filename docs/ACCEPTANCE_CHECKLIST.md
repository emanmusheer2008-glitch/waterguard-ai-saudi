# Manual acceptance checklist

Run on your own computer (about 30 minutes). In a terminal opened in the project folder:

```bat
cd %USERPROFILE%\OneDrive\Desktop\waterguard-ai-saudi
.venv\Scripts\activate
pip install -r requirements-dev.txt
python -m pytest -q
streamlit run app.py
```
Expected: pytest ends with `passed` and **0 failed**; the browser opens at http://localhost:8501.

## A. Benchmark demo (the verified evaluation)
| # | Check | Where | Pass if… |
|---|---|---|---|
| 1 | Overview loads | Overview | Precision 94.2%, Recall 38.2%, ROC-AUC 0.950, threshold 0.22; amber data disclaimer visible |
| 2 | Saudi vs L-Town distinction | Overview disclaimer; Model & Research → Dataset & Saudi context | Saudi = motivation (blue box), BattLeDIM = data (amber box) |
| 3 | Severe definition | Overview note | "≥ 40 m³/h … experimental threshold … not a standard" |
| 4 | Precision vs recall | Overview amber note | You can say: "When it alerts it's right 94% of the time, but it caught only 38% of severe steps." |
| 5 | Risk & Alerts (benchmark) | Risk & Alerts, source = Benchmark demo | Pick 6–23 Oct 2018; cards and chart update; click an alert row → explanation appears |
| 6 | Exploratory threshold | Risk & Alerts slider | Moving it shows the "exploratory" note; 0.22 is described as the real threshold |
| 7 | Detection timeline | Risk & Alerts → Detection timeline | Leakage (top) with red alert dots and 40 m³/h line; risk (bottom) with 0.22 line |
| 8 | Inspection guidance | Inspection & Sensors | Network map loads; "Any timestamp" on 10 Oct 2018 → toggle shows a star at the true leak |
| 9 | Sensor intelligence | Inspection & Sensors → Sensor intelligence | Two importance views; the n215 explanation is shown |
| 10 | Model & Research | all 5 tabs | Confusion matrix TN 22,332 · FP 210 · FN 5,561 · TP 3,433; limitations and disclaimer present |
| 11 | Not live | page headers | Benchmark pages show "Retrospective replay … Not live monitoring." |

## B. Analyze Data (your own data)
| # | Check | Pass if… |
|---|---|---|
| 12 | Sample file | Analyze Data → **Use the sample file** → validation shows *Ready to analyse*, 1,152 observations, 04 Oct 2026 – 07 Oct 2026, 5 min, Compatible |
| 13 | Run | **Run analysis** → 258 alerts, highest risk 0.753 on 06 Oct 2026 18:40, 6 reduced-context rows |
| 14 | No labels needed | The ground-truth card says "no labels for this data"; nothing asks for labels |
| 15 | Alert explanation | Click an alert row → sensitivity chart, signal table and inspection ranking appear |
| 16 | Export | **Predictions (CSV)** downloads with columns Timestamp, risk_probability, alert, reduced_context, imputed_values |
| 17 | Session carries over | Risk & Alerts and Inspection & Sensors open on **Your analysis** without re-uploading; the header pill names the sample file |
| 18 | Optional labels | Expand "Optional: ground-truth labels" → tick "Use the sample labels" → Run analysis → "Evaluation against your labels" cards appear, labelled evaluation only |
| 19 | Bad file | Upload any unrelated CSV (e.g. a spreadsheet export) → red/clear validation errors (e.g. missing Timestamp or missing sensor columns); Run analysis is disabled |
| 20 | Template | Download **Blank template** → it opens in Excel with Timestamp + 119 columns |
| 21 | Separate files | (optional) Input format → *Separate files per sensor group* → four upload boxes appear |
| 22 | Data Guide | Data Guide page explains formats, validation rules and output columns |
| 23 | Clean terminal | No red tracebacks or deprecation warnings in the Streamlit window while clicking through everything |

## C. API (optional)
```bat
uvicorn api.main:app --port 8000
```
| # | Check | Pass if… |
|---|---|---|
| 24 | Health | http://localhost:8000/health → `{"status":"ok","model_loaded":true,...}` |
| 25 | Docs | http://localhost:8000/docs → interactive page; try **POST /analyze** with `samples/waterguard_sample_ltown2018_shifted_to_2026.csv` → `"n_alerts": 258` |

If anything fails, copy the error text and bring it back to Claude.
