"""6. Data Guide — how to prepare data for Analyze Data, what the outputs mean, and what other networks need."""
from pathlib import Path

import pandas as pd
import streamlit as st

from ui.common import bundle, hero, link_to, note, section
from waterguard import service as svc
from waterguard.schema import GROUP_PREFIX, GROUP_UNITS
from waterguard.wording import WORDING

ROOT = Path(__file__).resolve().parents[2]


def render():
    hero("Data Guide", "How to prepare a file for Analyze Data, what every output column means, and what a different "
         "network would need.", badge=("neutral", "Reference · applies to Analyze Data and the API"))
    sch = svc.schema_description(bundle())

    section("1 · What you need")
    st.markdown(f"""
- **SCADA measurements from the L-Town benchmark network** (BattLeDIM 2018 or 2019 files, or data generated with the
  L-Town model) at a **5-minute** interval. Any year is accepted.
- **No leakage labels are needed.** If you have them, upload them separately for evaluation only.
- Up to **{sch['limits']['max_rows']:,} rows** (about one year) and **{sch['limits']['max_file_mb']} MB**; at least
  {sch['limits']['min_rows']} rows. The first 30 minutes are flagged *reduced context*.
""")
    note(f"<b>Different network?</b> {WORDING['different_network']}", "gt")

    section("2 · Accepted formats")
    st.dataframe(pd.DataFrame([
        {"Format": "One wide CSV", "Layout": "Timestamp + 119 prefixed columns (P_, D_, F_, L_)",
         "Notes": "Comma-separated, or ';' with decimal commas. Column order does not matter."},
        {"Format": "One wide XLSX", "Layout": "Same columns on the first sheet", "Notes": "Only the first sheet is read."},
        {"Format": "BattLeDIM workbook (XLSX)", "Layout": "Sheets 'Pressures (m)', 'Demands (L_h)', 'Flows (m3_h)', 'Levels (m)'",
         "Notes": "e.g. 2018_SCADA.xlsx or 2019_SCADA.xlsx, as downloaded."},
        {"Format": "Four separate files", "Layout": "One per group, Timestamp + unprefixed sensor IDs",
         "Notes": "e.g. 2019_SCADA_Pressures.csv … Levels.csv; choose 'Separate files per sensor group'."},
    ]), width="stretch", hide_index=True)

    section("3 · Column schema")
    for g, v in sch["column_groups"].items():
        with st.expander(f"{g.title()} — {v['count']} columns, prefix {v['prefix']}, unit {v['unit']}"):
            st.code(", ".join(v["columns"]), language=None, wrap_lines=True)
    st.caption("Prefixes are required in the wide format because some nodes have both a pressure sensor and a demand "
               "meter (e.g. P_n1 and D_n1). Extra columns are ignored; columns that look like leakage labels are removed "
               "and reported.")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Blank template (CSV)", svc.template_csv(bundle()), file_name="waterguard_input_template.csv",
                       mime="text/csv", icon=":material/description:", width="stretch")
    c2.download_button("Sample data (CSV)", (ROOT / "samples" / "waterguard_sample_ltown2018_shifted_to_2026.csv").read_bytes(),
                       file_name="waterguard_sample_ltown2018_shifted_to_2026.csv", mime="text/csv",
                       icon=":material/download:", width="stretch")
    with c3:
        link_to("analyze", "Go to Analyze Data", ":material/upload_file:")

    section("4 · What validation checks")
    st.dataframe(pd.DataFrame([
        ("File", "Type (.csv/.xlsx), size, readable workbook/CSV", "Error"),
        ("Timestamp", "Column present, every value a date; time zones converted to UTC", "Error / warning"),
        ("Columns", "All 119 sensor columns present; extras and label-like columns ignored", "Error / warning"),
        ("Order & duplicates", "Unsorted rows are sorted; duplicate timestamps are rejected", "Warning / error"),
        ("Sampling", "Median step must be 5 minutes and every step a multiple of 5 minutes; gaps are reported", "Error / warning"),
        ("Values", "Non-numeric values treated as missing; fully empty columns rejected; missing share reported", "Warning / error"),
        ("Compatibility", "Share of readings outside the range seen in training (≤5% compatible, ≤25% caution, else out of distribution)", "Warning"),
        ("Size", f"{sch['limits']['min_rows']}–{sch['limits']['max_rows']:,} rows", "Error"),
    ], columns=["Check", "Rule", "Outcome if it fails"]), width="stretch", hide_index=True)

    section("5 · Output columns (predictions CSV)")
    st.dataframe(pd.DataFrame([
        ("Timestamp", "The 5-minute timestamp from your file"),
        ("risk_probability", "Share of the 300 trees voting 'severe' (0–1). A risk score, not a calibrated probability."),
        ("alert", "1 if risk_probability ≥ 0.22 (the validation-chosen threshold), else 0"),
        ("reduced_context", "1 if 5/30-minute change features lacked history (start of file or next to a gap)"),
        ("imputed_values", "1 if any sensor reading was missing and filled with the training median"),
        ("ground_truth_total_leak_m3h / ground_truth_severe", "Only if you uploaded labels; evaluation only, never used by the model"),
    ], columns=["Column", "Meaning"]), width="stretch", hide_index=True)

    section("6 · Frequently asked")
    with st.expander("Can I analyse 2026 data?"):
        st.markdown(WORDING["time_recency"])
    with st.expander("Can I analyse data from a Saudi utility?"):
        st.markdown(WORDING["different_network"] + " The validator will reject a different sensor layout, and flags data "
                    "whose values are far outside the L-Town training range.")
    with st.expander("Does uploading data retrain the model?"):
        st.markdown("No. The saved V2 model, imputer, feature order and 0.22 threshold are applied unchanged. Files are "
                    "processed in memory and not stored.")
    with st.expander("What do the explanations and inspection guidance mean?"):
        st.markdown(WORDING["importance"] + " " + WORDING["inspection_guidance"])
    with st.expander("Can I use this from my own software?"):
        st.markdown("Yes — the same service is exposed as an HTTP API (`uvicorn api.main:app`), with `POST /analyze` and "
                    "`POST /explain`. See `docs/API.md` in the repository.")
