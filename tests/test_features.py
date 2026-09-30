"""Feature generation, ground-truth exclusion and chronological split."""
import numpy as np
import pandas as pd
import pytest

from waterguard.features import build_features_v1, build_features_v2, feature_category, feature_columns
from waterguard.target import chronological_split

GT_WORDS = ("leak", "target", "severe", "p31", "p158", "p183", "p232", "p257", "p369", "p427",
            "p461", "p538", "p628", "p654", "p673", "p810", "p866")


def test_v2_has_60_features_and_no_month(synthetic_scada):
    X = build_features_v2(synthetic_scada)
    cols = feature_columns(X)
    assert len(cols) == 60
    assert "month" not in cols
    assert {"hour_sin", "hour_cos", "pressure_std", "F_p235", "tank_level"} <= set(cols)
    assert sum(c.startswith("P_") for c in cols) == 33


def test_v1_has_20_features_including_month(synthetic_scada):
    cols = feature_columns(build_features_v1(synthetic_scada))
    assert len(cols) == 20 and "month" in cols


@pytest.mark.parametrize("builder", [build_features_v1, build_features_v2])
def test_no_ground_truth_in_features(synthetic_scada, builder):
    for c in feature_columns(builder(synthetic_scada)):
        name = c.lower().removeprefix("p_").removeprefix("f_")
        assert not any(name == w or name.startswith(w + "_") or "leak" in name or "target" in name for w in GT_WORDS), c


def test_features_are_causal(synthetic_scada):
    """Changing a future row must not change any earlier feature row."""
    X1 = build_features_v2(synthetic_scada)
    mod = {k: v.copy() for k, v in synthetic_scada.items()}
    mod["pressures"].iloc[40:, 1:] += 10
    mod["flows"].iloc[40:, 1:] += 10
    X2 = build_features_v2(mod)
    pd.testing.assert_frame_equal(X1.iloc[:40], X2.iloc[:40])


def test_summary_features_are_correct(synthetic_scada):
    X = build_features_v2(synthetic_scada)
    P = synthetic_scada["pressures"].drop(columns="Timestamp")
    np.testing.assert_allclose(X["pressure_range"], P.max(axis=1) - P.min(axis=1))
    np.testing.assert_allclose(X["pressure_mean_diff_30m"].iloc[6:], (P.mean(axis=1).diff(6)).iloc[6:])
    np.testing.assert_allclose(X["hour_sin"] ** 2 + X["hour_cos"] ** 2, 1.0)


def test_chronological_split_order_and_no_overlap():
    df = pd.DataFrame({"Timestamp": pd.date_range("2018-01-01", periods=1000, freq="5min")[::-1], "x": range(1000)})
    tr, va, te = chronological_split(df)
    assert (len(tr), len(va), len(te)) == (600, 100, 300)
    assert tr["Timestamp"].max() < va["Timestamp"].min()
    assert va["Timestamp"].max() < te["Timestamp"].min()
    for s in (tr, va, te):
        assert s["Timestamp"].is_monotonic_increasing


def test_feature_categories():
    assert feature_category("P_n215") == "Individual pressure sensor"
    assert feature_category("pressure_std_diff_5m") == "Short-term change"
    assert feature_category("F_PUMP_1") == "Individual flow / pump"
