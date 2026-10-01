# Sample and template files

| File | What it is |
|---|---|
| `waterguard_input_template.csv` | Header row only: `Timestamp` plus the 119 required sensor columns in the expected order. Fill it with your own 5-minute data. |
| `waterguard_sample_ltown2018_shifted_to_2026.csv` | **BattLeDIM L-Town 2018 measurements (4–7 Oct 2018), with every timestamp shifted by exactly 2,922 days to 4–7 Oct 2026.** 1,152 rows. It demonstrates that the model is not tied to the calendar year. **These are not 2026 measurements and not Saudi data.** |
| `waterguard_sample_ground_truth_shifted_to_2026.csv` | The benchmark's total leakage (m³/h) for the same window, shifted the same way. **Optional, evaluation only**: upload it separately to compare alerts with labels. It is never used for prediction. |

The window covers the onset of the severe leak on pipe p158 (6 Oct 2018 02:35, here 6 Oct 2026 02:35), so the sample contains both normal and severe periods.

**Expected result** (verified by `tests/test_service.py`): validation status *ok*; 258 alerts; from the 7th row onward the probabilities are identical to the 2018 benchmark predictions for the same measurements. The first 6 rows are flagged *reduced context* because the 30-minute change features need 30 minutes of history.

Source data: Vrachimis et al. (2020), *Dataset of BattLeDIM*, Zenodo, https://doi.org/10.5281/zenodo.4017659, CC BY 4.0. The excerpt is redistributed with attribution; only the timestamps were changed.
