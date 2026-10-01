# WaterGuard AI Saudi — Learning Guide

This guide is written so you can **explain every part of WaterGuard yourself**: in an interview, an application essay or a viva. The language is simple; the content matches the final project exactly. Every number here comes from the executed code.

**Learn these ten first:** (1) the Saudi-motivation / benchmark-data distinction, (2) why "any leak > 0" fails, (3) why ≥ 40 m³/h and that it is experimental, (4) chronological split, (5) data leakage and how it was prevented, (6) precision vs recall, (7) why V1 had good ROC-AUC but useless recall, (8) threshold tuning on validation (0.22), (9) feature importance ≠ causation, plus the n215 story, (10) the limitations, especially the two test episodes.

---

## Part A — The water side

**SCADA** (*Supervisory Control and Data Acquisition*). The system a utility uses to collect sensor readings from its network and show them to operators. In WaterGuard every model input comes from the BattLeDIM SCADA file, one reading every 5 minutes for all of 2018.

**Pressure (metres of head).** How hard water pushes inside a pipe at a point. 33 sensors. A leak lets water escape, so pressure near it usually falls a little. That is why individual pressure sensors matter so much.

**Flow (m³/h).** How much water moves through a pipe per hour. Two inlet pipes (p227, p235) and one pump. Water lost through leaks has to be supplied, so inflow can rise, especially at night when normal use is low.

**Demand (L/h).** How much customers use, from 82 meters. It follows a daily rhythm (mornings, evenings). The model must not mistake normal morning demand for a leak, which is why time of day is a feature.

**Tank level (m).** Height of water in tank T1. Tanks buffer supply and demand. Unusual draining can hint at extra outflow.

**Leakage (m³/h).** The benchmark's ground truth: how much water each of 14 simulated leaks loses at each moment. It is only used to create labels and to grade the model, never as an input.

## Part B — The data-science side

**Time-series data.** Measurements ordered in time. Neighbouring points are strongly related: the pressure at 10:05 is almost the same as at 10:00. This affects how you split and evaluate.

**Feature.** One number the model looks at for each moment, e.g. "pressure at n229" or "spread of the 33 pressures" (`pressure_std`). V2 has 60.

**Feature engineering.** Building useful features from raw data, for example the 30-minute change in average pressure, or turning hour of day into `sin` and `cos` so that 23:55 and 00:00 end up close together instead of 24 hours apart.

**Target (label).** What the model tries to predict. Here: *is total leakage ≥ 40 m³/h at this 5-minute step?* (1 = severe, 0 = not).

**Class imbalance.** When one answer is much rarer than the other. About 17% of training steps are severe. A model can look "accurate" by always saying "not severe". WaterGuard counters this with class weights and by judging precision, recall and F1 instead of accuracy.

**Class weights.** Telling the model that mistakes on the rare class count more. `class_weight="balanced_subsample"` makes each tree weight severe and normal examples so that both classes matter equally.

**Chronological split.** Train on the past, choose settings on the next period, test on a later period:
- *Train* (1 Jan – 7 Aug): the model learns.
- *Validation* (8 Aug – 13 Sep): we make choices, here the alert threshold.
- *Test* (13 Sep – 31 Dec): used **once** at the end to measure honestly.

A random split would put near-identical neighbouring readings into both train and test, so the test would be a memory test, not a prediction test.

**Data leakage (the machine-learning kind).** Information the model would not have in real life sneaking into training or evaluation, which makes results look better than they are. WaterGuard checked for and prevented: ground truth as a feature, features that look into the future, preprocessing fitted on test data, and choosing the threshold on the test set. Automated tests check all of these.

**Random Forest.** Hundreds of decision trees. Each tree learns from a random sample of rows and considers a random subset of features at each split. Each tree votes, and the share of trees voting "severe" is the probability. It captures non-linear patterns and interactions between sensors without needing scaled inputs.

**Probability (risk score).** The model's output between 0 and 1, e.g. 0.73 ≈ "73% of trees say severe". Treat it as a **risk ranking score**. It is not calibrated, so 0.73 does not mean a 73% real-world chance.

**Classification threshold.** The cut-off that turns a probability into a yes/no alert. The default is 0.5. V2 uses **0.22**, chosen because it gave the best F1 on validation. Lower thresholds catch more severe periods but raise more false alarms.

**Confusion matrix.** The four possible outcomes:

| | Model: not severe | Model: severe (alert) |
|---|---|---|
| **Really not severe** | True Negative (TN) — correctly quiet | **False Positive (FP)** — false alarm |
| **Really severe** | **False Negative (FN)** — missed | True Positive (TP) — caught |

V2 on test: TN 22,332 · FP 210 · FN 5,561 · TP 3,433.

**Precision** = TP / (TP + FP) = 3,433 / 3,643 = **94.2%**. *When it alerts, how often is it right?*

**Recall** = TP / (TP + FN) = 3,433 / 8,994 = **38.2%**. *Of all severe moments, how many did it catch?*

> 94% precision does **not** mean 94% of leaks are detected. That is recall, and it is 38%.

**F1 score.** The harmonic mean of precision and recall = **0.543**. It is only high if both are high. For context, "always alert" already gets 0.444 on this test set.

**ROC-AUC** = **0.950**. Pick one random severe moment and one random normal moment: AUC is the chance the model gives the severe one the higher score. 0.5 is random, 1.0 is perfect. It measures **ranking** over all thresholds, not performance at the threshold you actually use.

**False positive / false negative in water terms.** A false positive sends a crew to look for a leak that isn't severe, which wastes time and erodes trust. A false negative leaves a severe loss running unnoticed. V2 makes few of the first and many of the second.

**Feature importance — two kinds.**
- *Impurity (MDI) importance*: how much each feature helped the trees split the **training** data. Fast, but can over-reward features that only helped on training episodes.
- *Permutation importance*: shuffle one feature on **held-out** data and see how much AUC drops. It answers "does the model need this signal on new data?"

In WaterGuard they disagree: `P_n215` is #1 by impurity but #54 of 60 by test permutation; `P_n229` is #1 by permutation. Know this story (Q24).

**Causation vs association.** Feature importance shows what the model **used** (association), not what **caused** a leak. A sensor can be important because it sits where a leak's effects show up, or by coincidence with one training episode, as with n215.

**Model artifact.** The saved trained model and everything needed to use it again: the Random Forest, the fitted imputer, the ordered feature list and the threshold (`models/*.joblib`).

**joblib.** A Python library for saving and loading such objects to disk (`joblib.dump`, `joblib.load`). Retraining V2 from scratch reproduces the saved probabilities to within 3e-16.

**Streamlit.** A Python library that turns a script into a web app. Each widget interaction reruns the script.

**Caching.** Because Streamlit reruns the script on every click, `@st.cache_data` keeps loaded data in memory so it is read once, not on every interaction. WaterGuard also never retrains or reads the 92 MB raw file in the app; it loads small pre-computed files (about 0.7 s).

**Deployment.** Putting the app on a public server so others can use it. Plan: push to GitHub → Streamlit Community Cloud (free) builds it from `requirements.txt` and serves `app.py`.

**Inspection guidance.** A separate, transparent heuristic: compare each pressure sensor with its usual value **at the same time of day** (from normal training periods), and rank the biggest drops. It suggests where to start looking. It is not leak localisation.

---

## Part C — The story in order
1. **Inspect the data.** The leakage CSV crashed with pandas defaults (`;` separators, `,` decimals) → read with `sep=";", decimal=","`.
2. **"Any leak > 0" is true 97.8% of the time** → useless target.
3. **Severity analysis** → experimental target: total leakage ≥ 40 m³/h (≈ 80th percentile).
4. **V1** → ROC-AUC 0.86 but recall 0.2% at threshold 0.5; `month` was its top feature.
5. **V2** → 60 hydraulic features, no month, a validation period, threshold 0.22 → precision 94%, recall 38%, AUC 0.95.
6. **Audit** → no data leakage. Found: only two test episodes, prevalence shift, Logistic Regression worse than random.
7. **Final pass** → everything reproduced from raw data; n215 importance story; inspection guidance checked against true leak locations.

## Part D — Strengths and limitations
**Strengths:** real public benchmark handled carefully; an honest time-based evaluation; a target backed by a data discovery; high-precision alerts and strong ranking; a dashboard that keeps model output and ground truth separate; reproducible, tested code.

**Limitations:** simulated network, one year, two severe test episodes; recall 38%; experimental threshold; non-stationary leak signatures; no true localisation; uncalibrated probabilities; correlated 5-minute steps.

---

## Part E — Interview questions and answers (31 here, 7 more in Part F)

1. **What problem does WaterGuard solve?**
   It flags periods when a water network's sensor patterns look like severe water loss, and suggests which pressure sensors to inspect first, so a human can prioritise. It is a decision-support prototype.

2. **Is this Saudi data?**
   No. It is the BattLeDIM L-Town benchmark, a simulated network from an international competition. Saudi Arabia is the motivation: MEWA's National Water Strategy highlights reducing network losses. The results say nothing about Saudi networks.

3. **Why is that distinction so important to you?**
   Because claiming Saudi data would be false, and anyone checking would lose trust in everything else. Motivation and evidence are different things.

4. **Why was "any leakage > 0" a bad target?**
   97.8% of timestamps already contain some leakage from small, long-running leaks. A label that is "yes" almost always teaches nothing, and a model that always says "yes" would look accurate.

5. **Why did you choose ≥ 40?**
   From the data: it is about the 80th percentile of 2018 total leakage, so it marks the upper fifth, the clearly severe periods.

6. **Is 40 an official engineering threshold?**
   No. It is an experimental modelling threshold I created for this prototype. It is not a utility, regulatory, competition or Saudi standard. I tested 30, 35, 37.56, 45 and 50 to show how results depend on it.

7. **Isn't choosing 40 from the full year a kind of leakage?**
   Partly a fair concern: the percentile used the whole year's labels. It affects only the label definition, not the features. Using the training-only 80th percentile (37.56) gives similar results (F1 0.58 vs 0.54).

8. **Why didn't you use a random train/test split?**
   Neighbouring 5-minute readings are almost identical, so a random split leaks near-copies into the test set. A real system predicts the future from the past, so I split by time.

9. **What is the validation set for?**
   Making choices — here, the alert threshold. If I chose it on the test set, the test score would be optimistic.

10. **Why did V1 fail?**
    At the default 0.5 threshold it raised only 47 alerts and caught 20 of 8,994 severe steps (recall 0.2%). Its top feature was the month, so it was partly memorising when 2018's leaks happened.

11. **How can ROC-AUC be good while recall is terrible?**
    ROC-AUC measures ranking across all thresholds. V1 ranked severe moments higher than normal ones but rarely above 0.5, so at that threshold it almost never alerted.

12. **Why did you remove month?**
    Leaks don't follow the calendar. Month let the model learn *when* leaks happened in 2018, which would not generalise to another year.

13. **Why tune the threshold at all?**
    0.5 is only a default. The right threshold depends on the trade-off you want between false alarms and misses, and it should be chosen on data the final test never sees.

14. **Why 0.22?**
    It gave the best F1 on the validation period (precision 0.31, recall 0.70, F1 0.43). I didn't look at test results when choosing it.

15. **What does 94% precision mean?**
    Of the 3,643 alerts V2 raised on the test period, 3,433 were during genuinely severe periods. Alerts are usually trustworthy.

16. **What does 38% recall mean?**
    Of 8,994 severe 5-minute steps, V2 caught 3,433 and missed 5,561. Most severe time went unflagged.

17. **Why is test precision (94%) so different from validation precision (31%)?**
    Mainly prevalence: severe steps were 13% of validation but 28.5% of test. With more positives around, the same alerts are more often right. Different leaks in each period also matter.

18. **How many leak episodes are in the test period?**
    Only two: pipe p158 (6–23 Oct) and pipe p369 (26 Oct–8 Nov). V2 alerted at the first 5-minute step of both, then covered 48% and 26% of them. Two episodes are too few to claim event-level reliability.

19. **Did you compare against baselines?**
    Yes, on the same split with thresholds chosen on validation. "Always alert": F1 0.44, AUC 0.5. Single signal: AUC 0.61. Logistic Regression: AUC 0.22. Isolation Forest: AUC 0.74. V2: AUC 0.95, precision 94%.

20. **Logistic Regression got AUC 0.22 — worse than random. What does that tell you?**
    The linear relationships it learned from training-period leaks reversed for test-period leaks. Leak signatures change over time, which is a warning about generalisation for any model here.

21. **Why a Random Forest?**
    It handles non-linear relationships and sensor interactions, needs no scaling, copes with imbalance through class weights, and clearly beat the linear baseline. I didn't try many models to chase a number.

22. **How did you prevent data leakage?**
    Ground truth is never a feature; features use only current and past values; the imputer is fitted on training data only; the threshold is chosen on validation; the test set is scored once. Automated tests check each of these.

23. **Which features mattered most?**
    By held-out permutation importance, pressure sensor n229 by far, then n613, n188, n752 and n506. By training-time impurity importance, n215 and pressure spread lead. These features contributed strongly to the model's predictions; that doesn't mean they caused leaks.

24. **Why do the two importance rankings disagree?**
    n215 is almost constant (about 39.09 m). It dropped only during the March training leak, so the forest split on it heavily (impurity importance), but on later data shuffling it barely changes performance. Impurity importance is measured on training data; permutation importance on held-out data is the better guide.

25. **Can the model identify the exact leaking pipe?**
    No. It predicts *when* conditions look severe, not *where*. I added separate inspection guidance that ranks pressure sensors by their drop from normal. On the two test episodes, the sensor nearest the real leak ranked #1 and #2 of 33, which is encouraging but far from proof of localisation.

26. **How do you explain a single alert?**
    The Alert Explainer replaces one key signal at a time with its typical value and re-scores. That shows which signals pushed this alert's risk up. It describes the model's behaviour, not physical causes.

27. **Would you deploy this in a real Saudi network?**
    Not as it is. It was only tested on a simulated benchmark, recall is 38%, and leak patterns shift. A real deployment would need the utility's own data, engineers' validation, a pilot run alongside existing methods, and monitoring.

28. **What is the biggest limitation?**
    The evidence base: one simulated year with only two severe test episodes. Low recall comes second.

29. **How would you improve it?**
    Evaluate on BattLeDIM 2019 (an independent year); add alert logic that stays on for a whole episode (persistence/hysteresis) and measure event-level recall; use walk-forward validation; calibrate probabilities; try model-based localisation with the network model.

30. **How do you know your results are reproducible?**
    Retraining from raw data reproduces the saved probabilities to within 3e-16 and gives the same confusion matrix. Re-running the experiments produced byte-identical files, and 95 automated tests check the verified numbers, the pipeline, the API and the app.

31. **What did you learn most?**
    That a good metric can hide a useless system (V1), that framing the target matters as much as the model, that importance plots can mislead (n215), and that honest evaluation means reporting what doesn't work.

---

## Part F — The product side (Analyze Data, API, frontend)

**Training vs inference.** *Training* means learning from labelled history; it happened once, offline, in `src/train_v2.py`, and produced four saved artifacts: the Random Forest, the median imputer, the ordered list of 60 features and the 0.22 threshold. *Inference* means applying those saved artifacts to new data. When someone uploads a file, WaterGuard does inference only: no labels, no retraining, the same feature code. We proved the inference path is identical: run on the 2018 data, it reproduces the benchmark test probabilities to within 2.2e-16.

**Benchmark demo vs your analysis.** The *benchmark demo* is the verified 2018 evaluation, where ground truth exists, so you can see hits and misses. *Your analysis* is an uploaded file; there is usually no ground truth, so the app shows only model output (unless you upload labels separately, for evaluation only). The two are never mixed on screen.

**2018 vs 2026 timestamps.** The model has no calendar feature, so the year doesn't matter. The sample file is real 2018 L-Town data with every timestamp moved to 2026, and from the 7th row it produces exactly the same risk as the benchmark. The first 6 rows (30 minutes) are flagged *reduced context* because the 30-minute change features need history.

**Domain shift (network compatibility).** A model learns the "normal" of one network: its sensors, pipes, pressures and daily patterns. A different network, for example a Saudi city's, has different sensors and different normal values, so the L-Town model's rules don't transfer. That is why the upload validator requires the exact L-Town sensor columns and checks whether values fall within the training range. Recent data from another network is *not* compatible just because it is recent. A new network needs its own labelled history, retraining, chronological validation and its own threshold.

**Validation of inputs.** Before any prediction, the file is checked: timestamp column, all 119 sensors, 5-minute sampling, duplicates, gaps, missing values, value ranges. Problems are reported with clear codes (e.g. `MISSING_COLUMNS`, `INVALID_SAMPLING`) instead of silently producing wrong numbers.

**Service layer and API.** All ML logic lives in one Python module (`service.py`). The Streamlit app and the FastAPI API both call it, so there is one source of truth. An API lets any other program (e.g. a future React/Next.js website) send a file over HTTP and get JSON back. The frontend never re-implements the model in JavaScript.

**Real unseen-year check.** The official BattLeDIM 2019 workbook was uploaded through the same pipeline. It validated as compatible and was analysed in about 1.5 minutes. With 2019 labels, precision was 0.998 and recall 0.678, *but* 2019 has far more leakage: 82% of it counts as severe under the 2018-based 40 m³/h rule, so even "always alert" would score 82% precision. Lesson: a threshold defined from one period's distribution can lose meaning in another, so always check prevalence before quoting metrics.

### More interview questions

32. **Can I upload new data and get predictions?**
    Yes, if it is L-Town-compatible SCADA at 5-minute steps. WaterGuard validates it, builds the same 60 features, applies the saved model and threshold, and returns risk, alerts, explanations, inspection guidance and a CSV. No labels needed, no retraining.

33. **Does it work on 2026 data?**
    Yes for L-Town data: no calendar feature is used, and shifted data gives identical predictions. But a 2026 file from a *different* network is not valid input.

34. **Could a Saudi utility upload its SCADA data?**
    Not into this model. Its sensors and hydraulics differ. It would need its own historical data, verified leak records, retraining, chronological validation and threshold selection. The validator rejects a different sensor layout and flags out-of-range values.

35. **How do you make sure uploaded ground truth can't leak into predictions?**
    Only the 119 schema columns survive validation; label-like columns are removed and reported. Labels can only arrive as a separate file, which is joined to predictions after inference. A test adds fake label columns and checks that predictions are unchanged.

36. **Why separate the service layer from Streamlit?**
    So the same tested code serves the app and the API, and a future web frontend can replace Streamlit without touching the ML. Duplicated model logic would eventually drift apart.

37. **What did testing on 2019 data teach you?**
    The pipeline handles a real, unseen, full-year file. It also showed that metrics depend on prevalence: with 82% of 2019 labelled severe, 99.8% precision is far less impressive than it sounds.

38. **What happens if a file has gaps or missing values?**
    Gaps are reported and no prediction is made for missing steps; change features next to a gap are filled from training medians and flagged *reduced context*. Missing readings are imputed with training medians and flagged *imputed*, so the user knows which predictions are lower confidence.
