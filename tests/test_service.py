"""Inference service: validation, feature pipeline, inference, explanations, ground-truth handling."""
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from conftest import ROOT, needs_models, needs_scada
from waterguard import service as svc
from waterguard.config import OUTPUTS_DIR

SAMPLE = ROOT / "samples" / "waterguard_sample_ltown2018_shifted_to_2026.csv"
SAMPLE_GT = ROOT / "samples" / "waterguard_sample_ground_truth_shifted_to_2026.csv"
OFFSET = pd.Timestamp("2026-10-04") - pd.Timestamp("2018-10-04")  # 2,922 days

pytestmark = needs_models


@pytest.fixture(scope="module")
def bundle():
    return svc.load_bundle()


@pytest.fixture(scope="module")
def sample_bytes():
    return SAMPLE.read_bytes()


@pytest.fixture
def sample_df(sample_bytes):
    df = svc.read_table(sample_bytes, "sample.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


@pytest.fixture(scope="module")
def sample_result(sample_bytes, bundle):
    return svc.analyze(sample_bytes, "sample.csv", bundle=bundle)


def codes(rep):
    return {i.code for i in rep.issues}


def to_bytes(df):
    return df.to_csv(index=False).encode()


# ------------------------------------------------------------------ happy path
def test_model_bundle_loads(bundle):
    assert len(bundle.features) == 60 and "month" not in bundle.features
    assert round(bundle.threshold, 2) == 0.22
    assert bundle.model.n_features_in_ == 60
    assert len(bundle.columns) == 120  # Timestamp + 119


def test_valid_sample_analyses_without_ground_truth(sample_result):
    r = sample_result
    assert r.ok and r.report.status == "ok" and r.report.issues == []
    assert len(r.results) == 1152 and r.summary["n_alerts"] == 258
    assert r.evaluation is None and r.ground_truth is None
    assert r.report.interval_minutes == 5 and r.report.compatibility["level"] == "compatible"


def test_2026_timestamps_give_identical_predictions_to_2018_benchmark(sample_result):
    """Same measurements, timestamps shifted 8 years: identical risk after the 30-minute warm-up."""
    res = sample_result.results.copy()
    assert res["Timestamp"].min() == pd.Timestamp("2026-10-04 00:00")
    res["Timestamp"] -= OFFSET
    bench = pd.read_csv(OUTPUTS_DIR / "predictions_v2.csv", parse_dates=["Timestamp"])
    m = res.merge(bench, on="Timestamp")
    assert len(m) == 1152
    d = np.abs(m["risk_probability_x"] - m["risk_probability_y"]).to_numpy()
    assert d[6:].max() < 1e-12
    assert (m["alert"].iloc[6:].astype(int).to_numpy() == m["prediction"].iloc[6:].to_numpy()).all()
    assert sample_result.flags["reduced_context"].iloc[:6].all() and not sample_result.flags["reduced_context"].iloc[6:].any()


def test_probability_range_and_threshold_application(sample_result, bundle):
    p = sample_result.results["risk_probability"]
    assert p.between(0, 1).all()
    assert (sample_result.results["alert"] == (p >= bundle.threshold)).all()


def test_feature_order_matches_saved_model(sample_result, bundle):
    assert [c for c in sample_result.X.columns if c != "Timestamp"] == bundle.features
    assert sample_result.X_imputed.shape == (1152, 60)


def test_column_order_in_upload_does_not_matter(sample_df, sample_result):
    shuffled = sample_df[["Timestamp"] + list(sample_df.columns[1:][::-1])]
    r = svc.analyze_frame(shuffled)
    np.testing.assert_allclose(r.results["risk_probability"], sample_result.results["risk_probability"], atol=1e-12)


# ------------------------------------------------------------------ ground truth
def test_ground_truth_columns_in_upload_are_excluded(sample_df, sample_result):
    df = sample_df.copy()
    df["p158"] = 99.0
    df["total_leak"] = 99.0
    df["target"] = 1
    r = svc.analyze_frame(df)
    assert {"p158", "total_leak", "target"} <= set(r.report.ground_truth_like_columns)
    assert "GROUND_TRUTH_COLUMNS_IGNORED" in codes(r.report)
    assert not {"p158", "total_leak", "target"} & set(r.clean.columns)
    np.testing.assert_allclose(r.results["risk_probability"], sample_result.results["risk_probability"], atol=1e-12)


def test_separate_ground_truth_is_evaluation_only(sample_bytes, sample_result):
    r = svc.analyze(sample_bytes, "s.csv", ground_truth=(SAMPLE_GT.read_bytes(), "gt.csv"))
    np.testing.assert_allclose(r.results["risk_probability"], sample_result.results["risk_probability"], atol=1e-12)
    ev = r.evaluation
    assert ev["matched_rows"] == 1152 and 0 <= ev["precision"] <= 1 and ev["tp"] + ev["fn"] > 0
    out = r.predictions_frame()
    assert {"ground_truth_total_leak_m3h", "ground_truth_severe"} <= set(out.columns)


def test_ground_truth_with_no_matching_timestamps():
    gt = pd.DataFrame({"Timestamp": pd.date_range("2030-01-01", periods=3, freq="5min"), "total_leak": [1, 50, 2]})
    res = pd.DataFrame({"Timestamp": pd.date_range("2026-01-01", periods=3, freq="5min"),
                        "risk_probability": [.1, .5, .2], "alert": [False, True, False]})
    assert svc.evaluate_against_ground_truth(res, gt)["matched_rows"] == 0


# ------------------------------------------------------------------ validation failures
def test_missing_timestamp(sample_df):
    rep, clean = svc.validate_input(sample_df.drop(columns="Timestamp"))
    assert clean is None and "MISSING_TIMESTAMP" in codes(rep)


def test_bad_timestamp_values(sample_df):
    df = sample_df.copy().astype({"Timestamp": str})
    df.loc[5, "Timestamp"] = "not a date"
    rep, clean = svc.validate_input(df)
    assert clean is None and "BAD_TIMESTAMP" in codes(rep)


def test_missing_sensors(sample_df):
    rep, clean = svc.validate_input(sample_df.drop(columns=["P_n215", "F_p235", "D_n1"]))
    assert clean is None and "MISSING_COLUMNS" in codes(rep)
    issue = next(i for i in rep.issues if i.code == "MISSING_COLUMNS")
    assert set(issue.details["missing"]) == {"P_n215", "F_p235", "D_n1"}


def test_unprefixed_columns_get_a_hint(sample_df):
    df = sample_df.rename(columns={"P_n215": "n215"})
    rep, _ = svc.validate_input(df)
    assert "prefix" in next(i for i in rep.issues if i.code == "MISSING_COLUMNS").message


def test_extra_columns_are_ignored(sample_df, sample_result):
    df = sample_df.copy()
    df["operator_note"] = "ok"
    df["P_n9999"] = 1.0
    r = svc.analyze_frame(df)
    assert "EXTRA_COLUMNS" in codes(r.report) and r.ok
    np.testing.assert_allclose(r.results["risk_probability"], sample_result.results["risk_probability"], atol=1e-12)


def test_duplicate_timestamps_rejected(sample_df):
    df = pd.concat([sample_df, sample_df.iloc[[10]]])
    rep, clean = svc.validate_input(df)
    assert clean is None and "DUPLICATE_TIMESTAMPS" in codes(rep)


def test_unsorted_timestamps_are_sorted(sample_df, sample_result):
    r = svc.analyze_frame(sample_df.sample(frac=1, random_state=0))
    assert "UNSORTED_TIMESTAMPS" in codes(r.report) and r.ok
    np.testing.assert_allclose(r.results["risk_probability"], sample_result.results["risk_probability"], atol=1e-12)


def test_missing_values_are_imputed_and_flagged(sample_df):
    df = sample_df.copy()
    df.loc[100:109, "P_n229"] = np.nan
    df.loc[200, "F_p235"] = np.nan
    r = svc.analyze_frame(df)
    assert r.ok and "MISSING_VALUES" in codes(r.report)
    assert r.flags["imputed_values"].iloc[100:110].all() and r.flags["imputed_values"].iloc[200]
    assert r.results["risk_probability"].between(0, 1).all()


def test_fully_empty_column_rejected(sample_df):
    df = sample_df.copy()
    df["P_n229"] = np.nan
    rep, clean = svc.validate_input(df)
    assert clean is None and "EMPTY_COLUMNS" in codes(rep)


def test_non_numeric_values(sample_df):
    df = sample_df.astype({"P_n1": object})
    df.loc[3, "P_n1"] = "sensor fault"
    r = svc.analyze_frame(df)
    assert r.ok and "NON_NUMERIC_VALUES" in codes(r.report) and r.flags["imputed_values"].iloc[3]


def test_invalid_sampling_interval(sample_df):
    rep, clean = svc.validate_input(sample_df.iloc[::2])  # 10-minute data
    assert clean is None and "INVALID_SAMPLING" in codes(rep)


def test_off_grid_timestamps(sample_df):
    df = sample_df.copy()
    df.loc[50, "Timestamp"] += pd.Timedelta("2min")
    rep, clean = svc.validate_input(df)
    assert clean is None and "IRREGULAR_SAMPLING" in codes(rep)


def test_gaps_no_prediction_for_missing_steps_and_context_flagged(sample_df):
    df = sample_df.drop(index=range(300, 312))  # one hour missing
    r = svc.analyze_frame(df)
    assert r.ok and "GAPS" in codes(r.report) and r.report.missing_steps_in_gaps == 12
    assert len(r.results) == len(df)
    assert r.flags["reduced_context"].iloc[300]  # first row after the gap lacks history


def test_too_few_and_too_many_rows(sample_df, monkeypatch):
    rep, clean = svc.validate_input(sample_df.head(5))
    assert clean is None and "TOO_FEW_ROWS" in codes(rep)
    monkeypatch.setattr(svc, "MAX_ROWS", 100)
    rep, clean = svc.validate_input(sample_df)
    assert clean is None and "TOO_MANY_ROWS" in codes(rep)


def test_timezone_aware_timestamps(sample_df):
    df = sample_df.copy()
    df["Timestamp"] = df["Timestamp"].dt.tz_localize("Asia/Riyadh")
    r = svc.analyze_frame(df)
    assert r.ok and "TIMEZONE_REMOVED" in codes(r.report)


def test_out_of_distribution_values_are_flagged(sample_df):
    df = sample_df.copy()
    p = [c for c in df.columns if c.startswith("P_")]
    df[p] = df[p] * 3 + 40  # a different pressure regime
    r = svc.analyze_frame(df)
    assert r.report.compatibility["level"] == "out_of_distribution"
    assert "VALUES_OUTSIDE_TRAINING_RANGE" in codes(r.report)


# ------------------------------------------------------------------ file reading
def test_unsupported_and_empty_files():
    for content, name, code in [(b"x", "data.txt", "UNSUPPORTED_FORMAT"), (b"", "data.csv", "EMPTY_FILE"),
                                (b"x", "data.xlsx.exe", "UNSUPPORTED_FORMAT")]:
        with pytest.raises(svc.InputError) as e:
            svc.read_table(content, name)
        assert e.value.code == code


def test_file_too_large(monkeypatch):
    monkeypatch.setattr(svc, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(svc.InputError) as e:
        svc.read_table(b"Timestamp,a\n2026-01-01,1\n", "x.csv")
    assert e.value.code == "FILE_TOO_LARGE"


@pytest.mark.parametrize("content", [b"Timestamp,a\n2026-01-01 00:00,1,2,3\n",      # every row too long
                                     b"Timestamp,a\n2026-01-01 00:00,1\n2026-01-01 00:05,1,2\n",  # ragged
                                     b"\xff\xfe\x00\x00garbage\x00\n\x81\x82"])          # not text
def test_malformed_csv(content):
    with pytest.raises(svc.InputError) as e:
        svc.read_table(content, "bad.csv")
    assert e.value.code == "MALFORMED_CSV"


def test_malformed_xlsx():
    with pytest.raises(svc.InputError) as e:
        svc.read_table(b"this is not a workbook", "bad.xlsx")
    assert e.value.code == "MALFORMED_XLSX"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("hello.txt", "zip but not xlsx")
    with pytest.raises(svc.InputError) as e:
        svc.read_table(buf.getvalue(), "bad.xlsx")
    assert e.value.code == "MALFORMED_XLSX"


def test_semicolon_decimal_comma_csv(sample_df, sample_result):
    content = sample_df.to_csv(index=False, sep=";", decimal=",").encode()
    r = svc.analyze(content, "eu.csv")
    np.testing.assert_allclose(r.results["risk_probability"], sample_result.results["risk_probability"], atol=1e-12)


def test_wide_xlsx_and_battledim_workbook(sample_df, sample_result, bundle):
    buf = io.BytesIO()
    sample_df.head(60).to_excel(buf, index=False)
    r = svc.analyze(buf.getvalue(), "wide.xlsx")
    assert r.ok and len(r.results) == 60
    np.testing.assert_allclose(r.results["risk_probability"], sample_result.results["risk_probability"].iloc[:60], atol=1e-12)
    from waterguard.schema import SHEET_NAMES, wide_to_scada
    scada = wide_to_scada(sample_df.head(60), bundle.schema)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as w:
        for g, name in SHEET_NAMES.items():
            scada[g].to_excel(w, sheet_name=name, index=False)
    r2 = svc.analyze(buf.getvalue(), "2026_SCADA.xlsx")
    np.testing.assert_allclose(r2.results["risk_probability"], r.results["risk_probability"], atol=1e-12)


def test_combine_group_files(sample_df, sample_result, bundle):
    from waterguard.schema import wide_to_scada
    scada = wide_to_scada(sample_df, bundle.schema)
    files = {g: (df.to_csv(index=False, sep=";", decimal=",").encode(), f"2026_SCADA_{g}.csv") for g, df in scada.items()}
    wide = svc.combine_group_files(files)
    r = svc.analyze_frame(wide)
    np.testing.assert_allclose(r.results["risk_probability"], sample_result.results["risk_probability"], atol=1e-12)
    with pytest.raises(svc.InputError) as e:
        svc.combine_group_files({g: files[g] for g in ("pressures", "flows")})
    assert e.value.code == "MISSING_GROUP_FILE"


# ------------------------------------------------------------------ explanations & guidance
def test_explanation_and_inspection(sample_result, bundle):
    ts = sample_result.results.loc[sample_result.results["risk_probability"].idxmax(), "Timestamp"]
    e = sample_result.explain(ts)
    assert e["alert"] and len(e["signals"]) == len(bundle.ref["explain_signals"])
    assert {"signal", "risk_change_if_typical", "current_value", "typical_median"} <= set(e["signals"][0])
    g = e["inspection"]
    assert len(g["ranking"]) == 8 and g["n_sensors"] == 33
    devs = [r["deviation_sigma"] for r in g["ranking"]]
    assert devs == sorted(devs)
    with pytest.raises(KeyError):
        sample_result.explain(pd.Timestamp("1999-01-01"))


def test_benchmark_explanations_reproduced_by_service(sample_result, bundle):
    """Service explanation for a benchmark timestamp equals the pre-computed dashboard file."""
    ts26 = pd.Timestamp("2026-10-06 12:00")
    ctx = pd.read_csv(OUTPUTS_DIR / "dashboard_context_v2.csv.gz", parse_dates=["Timestamp"])
    row = ctx[ctx["Timestamp"] == ts26 - OFFSET].iloc[0]
    for s in sample_result.explain(ts26)["signals"]:
        assert abs(s["risk_change_if_typical"] - row["sens_" + s["signal"]]) < 1e-4


def test_summary(sample_result):
    s = sample_result.summary
    assert s["n_observations"] == 1152 and sum(s["risk_histogram"]["counts"]) == 1152
    assert sum(p["steps"] for p in s["alert_periods"]) == s["n_alerts"]


# ------------------------------------------------------------------ full benchmark equivalence
@needs_scada
def test_service_reproduces_full_benchmark_test_predictions(bundle):
    from waterguard.config import DATA_DIR
    from waterguard.data import load_scada
    from waterguard.schema import scada_to_wide
    r = svc.analyze_frame(scada_to_wide(load_scada(cache_dir=DATA_DIR / "cache")), bundle=bundle)
    bench = pd.read_csv(OUTPUTS_DIR / "predictions_v2.csv", parse_dates=["Timestamp"])
    m = r.results.merge(bench, on="Timestamp")
    assert len(m) == 31_536
    assert np.abs(m["risk_probability_x"] - m["risk_probability_y"]).max() < 1e-12
    assert (m["alert"].astype(int) == m["prediction"]).all()


def test_european_csv_where_decimals_appear_late():
    """Regression: BattLeDIM 2019_Leakages.csv (';' + decimal commas, trailing ';') broke comma-first parsing."""
    rows = ["Timestamp;p1;p2;"] + [f"2026-01-01 00:{m:02d}:00;0;0;" for m in range(0, 30, 5)] + ["2026-01-01 00:30:00;6,79;1,5;"]
    df = svc.read_table("\n".join(rows).encode(), "2019_Leakages.csv")
    assert list(df.columns) == ["Timestamp", "p1", "p2"] and df["p1"].iloc[-1] == 6.79
    gt = svc.read_ground_truth("\n".join(rows).encode(), "2019_Leakages.csv")
    assert gt["total_leak"].iloc[-1] == pytest.approx(8.29)
