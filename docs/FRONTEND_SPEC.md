# WaterGuard AI Saudi — Frontend Specification

**Audience:** an AI frontend generator or a frontend developer building the production web interface.
**Rule zero:** do not invent scientific content, numbers or claims. Every number shown must come from the API (`docs/API.md`) or from the verified values in §J. Approved wording in §J must be used verbatim or shortened without strengthening it.

The current Streamlit app (`app.py`, `ui/`) is a working reference implementation of every page described here; screenshots are in `docs/screenshots/`.

---

## A. Product identity
- **Name:** WaterGuard AI Saudi
- **Positioning:** ML water-loss decision-support research prototype.
- **One-liner:** Flags severe water-loss periods from hydraulic SCADA data and suggests which pressure sensors to inspect first. Evaluated on the BattLeDIM L-Town benchmark.
- **Audience:** university admissions reviewers, technical reviewers, water-engineering students, recruiters. Not utility operators (it is not an operational system).
- **It is not:** a live monitoring system, a Saudi utility product, a leak locator, a SaaS with accounts or pricing.

## B. Design direction
**Feel:** precise, technical, clean, premium, credible, calm, modern, engineering-oriented. Think instrument panel and engineering report, not marketing site.

**Do**
- Strong typographic hierarchy. Suggested pairing: *IBM Plex Sans* (UI/body) + *IBM Plex Mono* (numbers, timestamps, sensor IDs). Tabular figures for all metrics.
- Structured whitespace on an 8 px grid; max content width ~1320 px.
- Subtle water-inspired language: deep blue ink, a single teal accent, fine hairlines, network-map line art. No literal water drops/waves as decoration.
- Charts are the hero visuals: thin lines, light grids, direct labels, consistent colours.
- Motion only to communicate state: upload progress, validation running, analysis running, panel/drawer open (150–200 ms ease-out). Respect `prefers-reduced-motion`.

**Don't:** heavy gradients, glowing orbs, purple "AI" aesthetic, glassmorphism, huge rounded cards everywhere, stock photos, emojis, fake live counters, "real-time" pulses, testimonials, pricing, login/signup, chatbot widgets, government-style crests or any Saudi agency branding/logos.

**Colour tokens** (light theme; provide dark theme by re-deriving, not inverting):
| Token | Hex | Use |
|---|---|---|
| ink | `#0F1B2D` | text, headings |
| muted | `#5B6B7F` | secondary text |
| line | `#E3E8EF` | borders, hairlines |
| surface-2 | `#F5F7FA` | panels, notes |
| primary | `#0B5FA5` | model output (risk line), primary buttons, benchmark mode |
| teal | `#0E8A83` | user-analysis mode, "typical" bands |
| sky | `#7FB8E6` | secondary series (validation) |
| alert | `#C8423B` | alerts, threshold line, "pressure below usual" |
| alert-bg | `rgba(200,66,59,.10)` | ground-truth severe bands |
| amber | `#C98A13` | ground truth / evaluation-only, cautions, experimental threshold line |
| ok | `#2E7D5B` | success, precision |

Semantic rule: **blue = model output, amber = ground truth (evaluation only), red = alert.** Keep this everywhere.

## C. Navigation
Top bar, left-aligned product name, then:

1. **Overview** (`/`)
2. **Analyze Data** (`/analyze`) — primary call to action, visually emphasised
3. **Risk & Alerts** (`/alerts`)
4. **Inspection & Sensors** (`/inspect`)
5. **Model & Research** (`/research`)

Secondary (right side of the top bar or footer): **Data Guide** (`/guide`), GitHub link (placeholder), API docs link.

**Global mode indicator.** Pages 3 and 4 show a segmented control **"Benchmark demo (2018) | Your analysis"** at the top, and every page header shows a status pill:
- Benchmark: blue pill — `Retrospective replay of the 2018 BattLeDIM benchmark. Not live monitoring.`
- User: teal pill — `Your analysis · {filename} · saved V2 model · not live monitoring`

The active analysis is held in client state (e.g. React context / Zustand) for the session so users don't re-upload per page. Never mix benchmark ground truth into user-mode views.

## D. Pages

### D1. Overview
**Purpose:** explain the product in 30 seconds; show the verified result honestly; route to the two modes.

| Section | Content |
|---|---|
| Hero | Title, positioning, one-liner. Neutral pill: "Research prototype · BattLeDIM L-Town benchmark · not live". |
| Data disclaimer (amber note) | `benchmark_disclaimer` |
| Two mode cards | **Benchmark demo** — "The verified evaluation: trained Jan–Aug 2018, threshold tuned Aug–Sep, scored once on Sep–Dec 2018." Button → `/alerts` (benchmark). **Analyze data** — "Upload L-Town-compatible SCADA data… no retraining, no labels needed." Button → `/analyze`. |
| Severe-period definition (note) | `experimental_threshold` |
| Headline metrics (4 cards) | Precision 94.2% · Recall 38.2% · ROC-AUC 0.950 · Alert threshold 0.22 — from `GET /benchmark` → `metrics` |
| Reading note (amber) | `precision` + `recall` wording + "That is why WaterGuard is decision support, not an autonomous leak detector." |
| Counts (4 cards) | Test observations 31,536 · Severe 8,994 · Detected 3,433 · False alerts 210 of 3,643 |
| Risk timeline | Chart CH1 (benchmark, hourly) |
| Episode table | From `benchmark.episodes`: period, duration (days), alert raised, delay to first alert (h), share alerted (bar), peak risk |
| How it works | One-line pipeline: SCADA → validation → 60 features → saved imputer → saved Random Forest → risk → alert if ≥ 0.22 → explanation & inspection guidance |
| Limitations | 6 bullets (see §J `limitations`) + responsible-use note |

States: loading = skeleton cards + chart placeholder; error (API down) = inline banner "Benchmark data unavailable — the API could not be reached." with retry.

### D2. Analyze Data (primary feature)
**Purpose:** upload → validate → run → results → export.

**Header note (amber, always visible):** `different_network` + `time_recency`.

**Step 1 · Provide data**
- Segmented control: **One file** | **Separate files per sensor group**.
- One file: drag-and-drop zone (`.csv`, `.xlsx`, ≤ 100 MB). Text: "One row per 5-minute timestamp, 119 prefixed sensor columns."
- Separate files: four drop zones — Pressures (m), Demands (L/h), Flows (m³/h), Levels (m) — each `Timestamp` + unprefixed IDs. Send these to `POST /analyze-groups` (fields `pressures`, `demands`, `flows`, `levels`). Do not reimplement validation in JS.
- Button **Use sample file** (fetches `/sample-files/sample`). Caption: "Real BattLeDIM L-Town measurements from 4–7 Oct 2018 with timestamps shifted to 2026. Not 2026 data, not Saudi data."
- Download buttons: Blank template, Sample data, Sample labels (evaluation only).
- Collapsible **Expected schema** — from `GET /sample-schema`: table of groups (prefix, count, unit, examples) + copyable full column list.
- Collapsible **Optional labels (evaluation only)** — second drop zone; copy: "Joined to predictions after inference and never used by the model."

**Step 2 · Validate** — call `POST /analyze` (it validates first). Show **Validation panel** (component C6):
- Six status tiles: Status (Ready / Ready with warnings / Cannot be analysed), Observations, Date range, Sampling interval (+ gap count), Missing readings (%), Network compatibility (Compatible / Use with caution / Not compatible + % outside training range).
- Issue list: one row per `validation.issues[]` with a text tag ERROR / WARNING / INFO (not colour alone) and the message.
- Preview (first 12 rows, Timestamp + a few columns) and missing share per group.

**Step 3 · Run analysis** — primary button; disabled when status = error. Loading state: progress bar with steps "Reading file → Validating → Building 60 features → Running saved model → Explaining top alerts" (large XLSX can take ~1.5 min; show elapsed time). The current API runs validate+infer in one call; the steps may be shown as an indeterminate sequence.

**Step 4 · Results** (active analysis)
- 5 summary cards: Observations (+ date range), Alerts (+ share), Highest risk (+ time), Alert periods, Lower-confidence rows (reduced context · imputed).
- Compatibility banner (red) if `compatibility.level != "compatible"`.
- Chart CH2 (risk timeline, user) with alert markers; ground-truth bands only if labels were uploaded.
- Chart CH3 (risk histogram) + Alert periods table (start, end, duration h, peak risk), sortable.
- Evaluation cards (only if labels): matched rows, precision, recall, severe share + amber note `evaluation.note`.
- **Alert table** (top 200 by risk): timestamp, risk bar, confidence (Normal/Lower). Selecting a row opens the **Alert detail drawer** (C9).
- Export: **Predictions CSV** (columns in §G), **Summary + validation JSON**. Links: "Open in Risk & Alerts", "Open in Inspection & Sensors".

Empty state: "Upload a file or use the sample file to begin. Files are processed in memory and not stored." Error states: file-level errors (400) shown as a red panel with code + message; validation failure (422) shows the validation panel with errors.

### D3. Risk & Alerts
**Purpose:** monitor risk, review alerts, explain one alert — one page, no hopping.
- Mode selector (benchmark / your analysis). User mode without analysis → empty state with button to Analyze Data.
- View toggle: **Monitor & alerts** | **Detection timeline**.
- *Monitor & alerts:* date-range picker; **what-if threshold slider** (0.05–0.80, default 0.22) with permanent caption "Exploratory only. The model's threshold is 0.22, chosen on validation data."; cards (observations, alerts, peak risk, alerts confirmed severe — benchmark/labels only, labelled evaluation only); CH2; alert table (row select) → alert detail panel below or drawer; "Download alerts in window (CSV)".
- *Detection timeline:* benchmark → CH4 (two stacked panels: ground-truth leakage with alert dots and 40 m³/h line; risk with 0.22 line). User → risk panel only (+ ground truth panel if labels uploaded).
- Benchmark data: `/benchmark/timeline?resolution=5min` (or `1h` for full-range overview), explanations from `/benchmark/explain`. User data: from the `/analyze` response; explanations for other timestamps via `/explain`.

### D4. Inspection & Sensors
- Mode selector; view toggle **Inspection guidance** | **Sensor intelligence**.
- *Inspection guidance:* top note `inspection_guidance`; observation picker (highest-risk alerts / any timestamp); 4 cards (observation, model status, sensors below −2σ "x of 33", "Start inspection near {sensor}"); **Network map** CH5; ranked table (rank, sensor, deviation σ, model reliance rank); benchmark only: toggle "Show true leak location (evaluation only)" (star marker) and the benchmark check table (`benchmark.inspection_check`) with note: "the sensor nearest the leaking pipe ranked #1 and #2 of 33 — two episodes on a simulated network are not evidence of localisation." User mode: note "No leak locations are known for uploaded data."
- *Sensor intelligence:* CH6 (top-15 impurity importance, coloured by category), CH7 (importance by category), CH8 (validation vs test permutation importance) with the n215 explanation (§J `n215`), signal explorer CH9, three short "what the signals mean" blocks (pressure, flow, tank/demand). Note `importance`.

### D5. Model & Research
Tabs: **Performance** · **Methodology** · **Dataset & Saudi context** · **Using other data** · **Limitations & responsible use**.
- Performance: 5 metric cards (ROC-AUC 0.9501, Precision 94.24%, Recall 38.17%, F1 0.5433, PR-AUC 0.881); confusion matrix (C13); interpretation bullets; ROC (CH10), PR (CH11), validation threshold curve (CH12); chronological split bar (CH13); baselines table; V1 → V2 table; severity sensitivity table.
- Methodology: research question, target, 8-step pipeline, training vs inference.
- Dataset & Saudi context: two side-by-side notes (blue: Saudi motivation; amber: benchmark data), dataset facts, sources with links.
- Using other data: 2018 vs 2026; time recency ≠ network compatibility; 8-step "different network" flow (stepper); future architecture labelled "does not exist today".
- Limitations & responsible use: bullets + `responsible_use`.

### D6. Data Guide (secondary)
Requirements, accepted formats table, column schema (expandable per group), template/sample downloads, validation rules table, output-column dictionary, FAQ (2026 data? Saudi utility data? does upload retrain? what do explanations mean? API?). Content in `ui/pages/data_guide.py`.

## E. Component inventory
| # | Component | Notes |
|---|---|---|
| C1 | Top navigation | Sticky; active state underline; collapses to menu < 768 px |
| C2 | Page header | Title, subtitle, **status pill** (mode) |
| C3 | Mode switch | Segmented control; keyboard arrows; announces change |
| C4 | Metric card | Label (uppercase, small), value (mono), note; optional top border tone: primary/ok/alert/amber |
| C5 | Note / callout | Variants: info (sky), evaluation-only (amber), error (red) |
| C6 | Validation panel | Status tiles + issue list with text tags + collapsible preview |
| C7 | File uploader | Drag-drop, type/size hint, progress, replace/remove, error text |
| C8 | Risk indicator | Probability value + bar with threshold tick at 0.22 + text "ALERT"/"No alert" |
| C9 | Alert detail drawer | Header (time, risk, status, ground truth or "no labels"), lower-confidence notes, sensitivity bar chart CH14, signal table, inspection top-8 |
| C10 | Alert table | Sortable, row selection, risk bar column, confidence column |
| C11 | Timeline chart | CH1/CH2/CH4 |
| C12 | Network map | CH5 |
| C13 | Confusion matrix | 2×2 with TN/FP/FN/TP labels and plain words |
| C14 | Performance charts | ROC, PR, threshold curve, split bar |
| C15 | Stepper | Analyze steps; "different network" flow |
| C16 | Research section | Prose blocks with headings, source list |
| C17 | Download button | Icon + file type |
| C18 | Empty / loading / error states | Skeletons, spinners with step text, retry |
| C19 | Footer | Disclaimer (`responsible_use`), dataset credit "BattLeDIM (Vrachimis et al., 2020), CC BY 4.0", author "Eman Musheer", GitHub placeholder |

## F. API contract
Full request/response examples, errors and limits: **`docs/API.md`**. Summary:

| Endpoint | Used by |
|---|---|
| `GET /health` | App start; show banner if `status != ok` |
| `GET /model-info` | Research page, footer version, wording |
| `GET /sample-schema`, `GET /sample-files/{template,sample,sample-ground-truth}` | Analyze Data, Data Guide |
| `POST /analyze` (multipart `file`, optional `ground_truth`; query `include_predictions`, `top_alerts`) | Analyze Data → active analysis |
| `POST /analyze-groups` (multipart `pressures`, `demands`, `flows`, `levels`, optional `ground_truth`) | Analyze Data, separate-files mode |
| `POST /explain` (multipart `file`, `timestamp`) | Alert drawer for user data beyond `top_alerts` |
| `GET /benchmark` | Overview, Research, Inspection check |
| `GET /benchmark/timeline?resolution=1h|5min` | Overview, Risk & Alerts (benchmark) |
| `GET /benchmark/explain?timestamp=` | Alert drawer + inspection (benchmark) |
| `GET /network` | Network map |

Error shape always `{error: {code, message, ...}}`; 400 = unreadable file, 422 = validation failed (includes `validation`), 404 = not found, 500 = generic. Show `message` to the user; never show raw JSON.

## G. Chart data contracts
| Chart | Type | Data | Encoding |
|---|---|---|---|
| CH1 Overview timeline | Line + bands | `/benchmark/timeline?resolution=1h`: `timestamp[]`, `risk_probability[]`, `ground_truth_severe[]` | x time; y risk 0–1; dashed red line at 0.22 labelled "alert threshold 0.22"; amber-red bands where `ground_truth_severe=1` labelled "ground truth, evaluation only" |
| CH2 Risk timeline | Line + markers | Benchmark: timeline `5min`; user: `predictions[]` (`timestamp`, `risk_probability`, `alert`) | As CH1; red dots at `alert=true`; bands only if labels exist |
| CH3 Risk histogram | Bar | `summary.risk_histogram.{bin_edges, counts}` | 10 bins 0–1; bins ≥ 0.2 in alert colour; caption explains |
| CH4 Detection timeline | 2 stacked panels, shared x | timeline `5min`: `ground_truth_total_leak_m3h`, `alert`, `risk_probability` | Top: leakage (m³/h) grey line, red dots at alerts, amber dashed line at 40 "experimental severity threshold"; bottom: risk, red dashed 0.22 |
| CH5 Network map | Line segments + scatter | `/network` links (x0,y0,x1,y1) + sensors (id,x,y); values = `inspection.all_deviations` | Equal aspect; pipes grey hairline; sensor colour diverging red(−4)…grey(0)…blue(+4) σ, size grows with drop; label top-5 IDs; optional amber star at true leak (benchmark, evaluation only); colour legend with text |
| CH6 Impurity importance | Horizontal bar | `/benchmark.feature_importance` sorted by `mdi_importance` (top 15) | Colour by category (see Streamlit `CAT_COLORS`) |
| CH7 Category importance | Horizontal bar | sum of `mdi_importance` by category | % labels |
| CH8 Permutation importance | Grouped horizontal bar | `validation_auc_drop_mean`, `test_auc_drop_mean` for union of top-8 MDI and top-8 test | Sky = validation, primary = test; x "ROC-AUC drop when shuffled" |
| CH9 Signal explorer | Line + band | Benchmark: needs a timeline of the chosen feature (add endpoint or ship `dashboard_context_v2.csv.gz` as static JSON); user: feature values are not returned by `/analyze` today — add if needed via the service (`AnalysisResult.X`) | Teal band = training non-severe 5–95%; dotted teal median |
| CH10 ROC | Line | `/benchmark.curves.roc.{x,y}` | Diagonal dotted "random" |
| CH11 PR | Line + point | `curves.pr.{x,y}`, point at (recall 0.3817, precision 0.9424) | Dotted baseline at prevalence 0.285 |
| CH12 Threshold selection | 3 lines | `/benchmark.threshold_validation[]` (`threshold, precision, recall, f1`) | Vertical dashed line at 0.22; title must say "validation only" |
| CH13 Split bar | Stacked horizontal | `/benchmark.splits.{train,validation,test}.{start,end,rows,severe_rate}` | Primary / amber / teal |
| CH14 Local sensitivity | Diverging horizontal bar | `signals[].{signal, risk_change_if_typical}` | Red > 0 (pushes risk up), teal < 0; zero line |

## H. Responsive behaviour
- **Desktop ≥ 1200 px:** layouts as described; drawers slide from the right (480 px).
- **Tablet 768–1199 px:** two-column grids become one/two columns; metric cards 2–3 per row; map and table stack; drawer becomes full-height sheet.
- **Mobile < 768 px:** single column; nav in a menu; metric cards 2 per row; tables become card lists (timestamp, risk bar, status) with horizontal scroll only inside tables; charts full width with reduced height (220–260 px), legends below; uploader full width; 16 px side gutter; no horizontal page scroll.

## I. Accessibility
- WCAG 2.2 AA contrast (text ≥ 4.5:1); base font 16 px, line-height 1.5.
- Every status uses text + colour: "ALERT"/"No alert", ERROR/WARNING/INFO tags, "Compatible / Use with caution / Not compatible".
- Charts: titles, axis labels with units, accessible name/description summarising the takeaway; provide a "view data table" toggle for each chart.
- Keyboard: all controls reachable; visible focus ring; table rows selectable with Enter/Space; drawer traps focus and closes with Esc; segmented controls use arrow keys.
- Uploader announces validation results via `aria-live="polite"`; loading states announced.
- Don't rely on hover for essential information; tooltips duplicate visible labels.
- Respect `prefers-reduced-motion` and `prefers-color-scheme`.

## J. Approved scientific wording
Source of truth: `src/waterguard/wording.py` (also returned as `wording` by `/model-info` and `/benchmark`). Use verbatim:

- **benchmark_disclaimer:** "WaterGuard is motivated by water loss in Saudi Arabia but is trained and evaluated on the BattLeDIM L-Town benchmark, a simulated international research network. None of the measurements come from Saudi Arabia or any Saudi utility."
- **saudi_motivation:** "Saudi Arabia's Ministry of Environment, Water and Agriculture (MEWA) identifies reducing losses in water networks as an improvement opportunity in its National Water Strategy, which estimates network losses at more than 25% in different regions. This is the motivation for WaterGuard, not evidence about its performance."
- **experimental_threshold:** "A severe water-loss period is a 5-minute step in which total benchmark leakage is at least 40 m³/h. This is an experimental threshold chosen for this prototype (about the 80th percentile of 2018 leakage). It is not an engineering, utility, regulatory, competition or Saudi standard."
- **precision:** "Precision 94.2%: of the 3,643 alerts raised on the held-out test period, 3,433 occurred during genuinely severe periods. When WaterGuard alerts, it is usually right."
- **recall:** "Recall 38.2%: WaterGuard caught 3,433 of 8,994 severe 5-minute steps and missed 5,561. Precision of 94% does not mean 94% of severe periods are detected."
- **alert_threshold:** "An alert is raised when the risk probability is at least 0.22. This threshold was chosen on the validation period only, never on the test period."
- **inspection_guidance:** "Inspection guidance, not leak localisation. Pressure sensors are ranked by how far they are below their usual level for the time of day. It suggests where to start looking; it does not identify a leaking pipe."
- **not_live:** "Retrospective replay of the 2018 BattLeDIM benchmark. Not live monitoring."
- **different_network:** "This model was trained on the simulated L-Town network. Data from a different network (different sensors, topology, pressures, demand patterns or operations) is not compatible and must not be analysed with it. A different network needs its own historical data, verified leak records, retraining, chronological validation and threshold selection."
- **time_recency:** "The model is not tied to the year 2018: compatible L-Town data with 2026 timestamps is processed the same way, because no calendar feature is used. Recent timestamps do not make data from another network compatible."
- **ground_truth:** "Ground truth (benchmark leakage) is shown for evaluation only. It is never a model input."
- **importance:** "Feature importance and explanations describe how the model uses signals. They do not show what caused a leak."
- **responsible_use:** "WaterGuard AI Saudi is an independent educational research prototype and is not affiliated with or endorsed by a Saudi government entity or water utility. It has not been validated on real Saudi network data and must not be used for operational decisions."
- **n215:** "P_n215 is #1 by training-time (impurity) importance but #54 of 60 on the test period. It reads almost constantly about 39.09 m and dropped only during the March training leak. P_n229 is the most relied-on signal on both validation and test."
- **limitations:** simulated network, one year; only two severe test episodes · recall 38% · experimental 40 m³/h threshold · says *when*, not *where* · L-Town-compatible data only · probabilities not calibrated.

**Verified numbers (never change):** precision 0.9424 · recall 0.3817 · F1 0.5433 · ROC-AUC 0.9501 · PR-AUC 0.8806 · threshold 0.22 · TN 22,332 · FP 210 · FN 5,561 · TP 3,433 · V1: precision 0.4255, recall 0.0022, F1 0.0044, ROC-AUC 0.8602.

**Forbidden phrasings:** "detects leaks with 94% accuracy", "real-time", "live network", "Saudi water data", "used by NWC/MEWA", "locates leaks", "AI-powered leak detection platform", "predicts leaks before they happen", any savings or ROI figure.

## K. Technology recommendation
Next.js (App Router) + React + TypeScript; Tailwind CSS with the tokens in §B; shadcn/ui (Radix) for accessible primitives (tabs, segmented control, dialog/drawer, table, tooltip); charts with Plotly.js (closest to the reference) or ECharts — keep one library; TanStack Query for API calls; Zustand (or React context) for the active analysis. The ML stays in Python: the frontend only calls the API. Deploy the API separately (e.g. a container on Render/Fly/Railway) and set `WATERGUARD_CORS_ORIGINS` to the frontend URL.
