# WaterGuard AI Saudi — Learning Guide

This guide is written so you can **explain every part of the project yourself**, in an interview or on a university application. The language is simple, but the technical content is accurate.

---

## 1. What problem does WaterGuard solve?
Water pipes leak. Some leaks are small and last for months. Others are big and appear suddenly. A utility can't dig up every pipe, but it does record sensor readings every few minutes. WaterGuard reads those readings and says: *"Right now the network looks like it did during periods of severe water loss."* That helps a human decide where to look first.

**Important honesty point:** the project is motivated by water loss in Saudi Arabia, but the data comes from **BattLeDIM L-Town**, a simulated international research benchmark. It is *not* Saudi data.

## 2. Key water-network terms
| Term | Simple meaning | Why it matters for leaks |
|---|---|---|
| **SCADA** | *Supervisory Control and Data Acquisition*: the system that collects sensor readings from a network and shows them to operators. | It is the source of all model inputs. |
| **Pressure** (metres of water head) | How hard water pushes inside the pipe at a point. | A leak lets water escape, so nearby pressure usually drops a little. |
| **Flow** (m³/h) | How much water moves through a pipe per hour. | Water lost through leaks must be replaced, so inflow can rise, especially at night when normal use is low. |
| **Demand** (L/h) | How much water customers use. | Normal demand changes over the day. The model must not confuse morning usage with a leak. |
| **Tank level** (m) | Height of water in a storage tank. | Tanks buffer supply and demand. Unusual draining can hint at extra outflow. |

## 3. Machine-learning terms
**Feature.** One number the model looks at for each moment in time, e.g. "pressure at sensor n215" or "spread of all pressures".

**Feature engineering.** Creating useful features from raw data. Examples from this project: the standard deviation across 33 pressure sensors (`pressure_std`); the 30-minute change in average pressure; turning the hour of day into `sin`/`cos` so that 23:55 and 00:00 end up close together.

**Random Forest.** Many decision trees, each trained on a random sample of the data and features. Each tree votes, and the share of "severe" votes becomes the probability. It handles non-linear patterns and interactions well.

**Classification probability.** The model outputs a number between 0 and 1, e.g. 0.73, meaning "73% of the trees voted severe". It is a *risk score*, not a guaranteed real-world probability (the scores are not calibrated).

**Threshold.** The cut-off that turns a probability into a yes/no alert. V2 raises an alert when probability ≥ 0.22. Lowering the threshold catches more severe periods but also raises more false alarms.

**Train / validation / test.**
- *Train*: the model learns from this data.
- *Validation*: used to make choices, here the threshold.
- *Test*: used **once**, at the end, to measure performance honestly.

**Why chronological splitting matters.** Sensor readings 5 minutes apart are almost identical. With a random split, the test set would contain near-copies of training rows, and the score would look unrealistically good. A real system always predicts the **future** from the **past**, so WaterGuard trains on January–early August, tunes on August–September, and tests on September–December.

## 4. Evaluation metrics
Take one test moment. There are four possible outcomes:

| | Model says "not severe" | Model says "severe" (alert) |
|---|---|---|
| **Really not severe** | True Negative (TN) | False Positive (FP): false alarm |
| **Really severe** | False Negative (FN): missed | True Positive (TP): caught |

This table is the **confusion matrix**. V2's version is TN 22,332 · FP 210 · FN 5,561 · TP 3,433.

- **Precision** = TP / (TP + FP) = 3,433 / 3,643 = **94.2%**. *When it alerts, how often is it right?*
- **Recall** = TP / (TP + FN) = 3,433 / 8,994 = **38.2%**. *Of all severe moments, how many did it catch?*
- **F1** is the harmonic mean of precision and recall = **0.543**. It stays high only if both are high.
- **ROC-AUC** = **0.950**. If you pick one random severe moment and one random normal moment, it is the chance the model gives the severe one a higher score. 0.5 is random guessing and 1.0 is perfect. It measures *ranking*, independent of the threshold.

**Feature importance.** How much each feature helped the forest split the data (mean decrease in impurity). It tells you what the *model used*. It does **not** tell you what *caused* a leak.

**Data leakage (the ML kind).** When information the model would not have in real life sneaks into training or evaluation. Examples: using the leakage ground truth as a feature, fitting preprocessing on test data, or choosing the threshold on the test set. WaterGuard avoids all three, and the tests check for them.

## 5. The story of the project
**Why "any leak > 0" did not work.** In 97.8% of timestamps, at least one leak is above zero, because several small leaks last almost all year. A target that is "yes" 98% of the time teaches the model nothing. So the project defines **severe** as *total leakage ≥ 40*. That is an experimental cut-off at roughly the 80th percentile, and it is **not** an official standard.

**Why V1 failed operationally despite a decent ROC-AUC (0.86).** V1 ranked risky moments reasonably well, but with the default threshold of 0.5 it raised only 47 alerts and caught 20 of 8,994 severe moments (recall 0.2%). Its top feature was `month`, so it was partly memorising *when* leaks happened in 2018. A good ranking is useless if the alert threshold never fires.

**What V2 changed.**
1. Kept all 33 individual pressure sensors.
2. Added flow, tank, time-of-day and short-term-change features.
3. Removed `month`.
4. Added a validation period and chose the threshold (0.22) there.

The result was precision 94%, recall 38% and ROC-AUC 0.95 on the untouched test period.

**What deeper analysis revealed.**
- The test period contains only **two** severe leak episodes. V2 alerted at the very start of both, but was only above threshold for part of each.
- An "always alert" rule gets F1 0.44, so V2's real advantage is its **precision and ranking**.
- Logistic Regression scored ROC-AUC 0.22, worse than random. The patterns of test-period leaks differ from training-period leaks, and a linear model could not adapt.

## 6. Strengths
- Real, public benchmark data, handled carefully (a tricky CSV format, integrity checks).
- An honest time-based evaluation with a separate validation period.
- A thoughtful target definition, backed by a data discovery.
- High-precision alerts (few false alarms) and strong ranking (AUC 0.95).
- A clear dashboard that separates model output from ground truth, with an explanation view.
- Reproducible: retraining gives identical probabilities, and automated tests guard the results.

## 7. Limitations
- A simulated network, one year, and only two severe test episodes.
- Many severe moments are missed (recall 38%).
- The model says *when* the network looks risky, not *where* the leak is.
- The severity threshold is experimental.
- Leak signatures change over time, so performance may not transfer.

---

## 8. Likely interview questions (with short answers)

1. **What does your project do, in one sentence?**
   It uses machine learning on water-network sensor data to flag periods when the network appears to be losing a lot of water, and shows this in a decision-support dashboard.

2. **Is this Saudi data?**
   No. It is the BattLeDIM L-Town international benchmark. The project is motivated by water loss in Saudi Arabia, but I evaluated it on public benchmark data.

3. **Why didn't you just predict "leak" vs "no leak"?**
   Because 97.8% of timestamps already contain some leakage from small persistent leaks. That label is almost always "yes", so I targeted severe total leakage instead.

4. **Where does the threshold of 40 come from?**
   From the data. It is about the 80th percentile of total 2018 leakage. It is an experimental benchmark definition, not an engineering standard. I also tested 30, 35, 45 and a training-only percentile to show how results change.

5. **Why didn't you shuffle the data before splitting?**
   Neighbouring 5-minute readings are nearly identical, so shuffling would leak information and inflate the score. A real system predicts the future from the past.

6. **What is the validation set for?**
   To choose the alert threshold. If I chose it on the test set, the test score would be optimistic.

7. **Your precision is 94% but recall is only 38%. Is that good?**
   It means alerts are trustworthy but many severe periods are missed. For an operator, few false alarms is valuable, but improving recall is the main future goal.

8. **Why is ROC-AUC high while recall is low?**
   ROC-AUC measures ranking across all thresholds. Recall depends on the one threshold chosen. The model ranks well, but at 0.22 it stays quiet during part of each episode.

9. **What was wrong with V1?**
   At the default threshold it hardly ever alerted (recall 0.2%), and its most important feature was the month, so it was partly learning calendar patterns rather than hydraulics.

10. **Why remove the month feature?**
    Because leaks don't care what month it is. Month let the model memorise when 2018's leaks happened, which would not generalise to a new year.

11. **Why a Random Forest?**
    It handles non-linear relationships and interactions between sensors, needs little scaling, and gives feature importances. A linear model (Logistic Regression) performed much worse in my comparison.

12. **Which features mattered most?**
    Individual pressure sensors such as n215, n229 and n114, and pressure spread (`pressure_std`, `pressure_range`). These features contributed strongly to the model's predictions. That doesn't prove they caused anything.

13. **How do you explain a single alert?**
    The Alert Explainer shows the signal values against typical training ranges, and how much the risk would change if each signal were replaced by its typical value. It is a descriptive view of the model, not a cause.

14. **How did you prevent data leakage?**
    Ground truth is never a feature, the imputer is fitted on training data only, features use only past values, the threshold is chosen on validation, and the test set is used once. Tests check these.

15. **What surprised you?**
    The whole test period contains only two severe leak episodes. Also, Logistic Regression was worse than random on test, which showed me that leak patterns change over time.

16. **How would you improve it?**
    Test on 2019 data, add alert logic that keeps an alarm on for the whole episode, localise leaks using the network model, and calibrate the probabilities.

17. **Could this be used by a utility today?**
    No. It is an educational prototype validated only on a simulated benchmark. Real use would need real data, validation with engineers and a proper deployment process.

18. **How do you know your results are reproducible?**
    Retraining with the same random seed reproduced the saved probabilities to within 2.2e-16, and automated tests check the verified numbers.

19. **What did you learn most from this project?**
    That a good metric can hide a weak system (V1), that problem framing and the target definition matter as much as the model, and that honest evaluation means reporting weaknesses too.

20. **What does F1 = 0.54 mean compared with a simple rule?**
    Always alerting would get F1 = 0.44 on this test set, so F1 alone is not impressive. The real gains are precision (94% vs 29%) and ranking (AUC 0.95 vs 0.5).
