"""Verified results, dashboard outputs and model artifacts."""
import joblib
import numpy as np
import pandas as pd

from conftest import needs_models
from waterguard.config import MODELS_DIR, OUTPUTS_DIR, V1_VERIFIED, V2_ALERT_THRESHOLD, V2_VERIFIED
from waterguard.dashboard_data import load_all, missing_outputs
from waterguard.evaluation import event_level_summary, point_metrics


def test_dashboard_required_outputs_exist():
    assert missing_outputs() == []


def test_dashboard_data_loads():
    d = load_all()
    assert len(d["pred"]) == len(d["ctx"]) == 31_536
    assert (d["pred"]["Timestamp"].values == d["ctx"]["Timestamp"].values).all()


def test_permutation_importance_output():
    p = pd.read_csv(OUTPUTS_DIR / "permutation_importance_v2.csv")
    assert len(p) == 60 and p["feature"].is_unique
    assert {"validation_auc_drop_mean", "test_auc_drop_mean", "mdi_importance"} <= set(p.columns)


def test_pressure_deviation_output_aligned_with_predictions():
    d = load_all()
    assert len(d["pdev"]) == len(d["pred"])
    assert (d["pdev"]["Timestamp"].values == d["pred"]["Timestamp"].values).all()
    assert d["pdev"].shape[1] == 34 and np.isfinite(d["pdev"].drop(columns="Timestamp").to_numpy()).all()
    assert set(d["pdev"].columns[1:]) == set(d["sensors"]["id"])


def test_v2_verified_results_unchanged():
    p = pd.read_csv(OUTPUTS_DIR / "predictions_v2.csv")
    m = point_metrics(p["target"], p["prediction"], p["risk_probability"])
    for k in ("tn", "fp", "fn", "tp"):
        assert m[k] == V2_VERIFIED[k]
    for k in ("precision", "recall", "f1", "roc_auc"):
        assert round(m[k], 4) == V2_VERIFIED[k]


def test_v2_predictions_consistent_with_threshold():
    p = pd.read_csv(OUTPUTS_DIR / "predictions_v2.csv")
    assert p["risk_probability"].between(0, 1).all()
    assert ((p["risk_probability"] >= V2_ALERT_THRESHOLD - 1e-9).astype(int) == p["prediction"]).all()


def test_v1_verified_results_unchanged():
    p = pd.read_csv(OUTPUTS_DIR / "predictions_v1.csv")
    m = point_metrics(p["severe_leak"], p["predicted_severe"], p["risk_probability"])
    for k, v in V1_VERIFIED.items():
        assert round(m[k], 4) == v


def test_threshold_chosen_on_validation_is_best_f1():
    tv = pd.read_csv(OUTPUTS_DIR / "threshold_validation_v2.csv")
    assert round(tv.loc[tv["f1"].idxmax(), "threshold"], 2) == V2_ALERT_THRESHOLD


def test_test_period_follows_validation():
    p = pd.read_csv(OUTPUTS_DIR / "predictions_v2.csv", parse_dates=["Timestamp"])
    assert p["Timestamp"].min() == pd.Timestamp("2018-09-13 12:00")
    assert p["Timestamp"].is_monotonic_increasing


def test_event_level_summary_logic():
    ts = pd.date_range("2018-01-01", periods=40, freq="5min")
    target = np.r_[np.zeros(10), np.ones(20), np.zeros(10)].astype(int)
    pred = np.zeros(40, int); pred[15] = 1
    df = pd.DataFrame({"Timestamp": ts, "target": target, "prediction": pred, "risk_probability": pred * 0.9, "total_leak": target * 45})
    ev = event_level_summary(df, min_steps=12)
    assert len(ev) == 1 and ev["detected"].iloc[0]
    assert ev["hours_to_first_alert"].iloc[0] == 5 * 5 / 60


@needs_models
def test_model_artifacts_load_and_predict_probabilities():
    model = joblib.load(MODELS_DIR / "waterguard_rf_v2.joblib")
    imputer = joblib.load(MODELS_DIR / "imputer_v2.joblib")
    feats = joblib.load(MODELS_DIR / "features_v2.joblib")
    thr = joblib.load(MODELS_DIR / "threshold_v2.joblib")
    assert len(feats) == 60 and "month" not in feats
    assert round(float(thr), 2) == V2_ALERT_THRESHOLD
    assert model.n_features_in_ == 60
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(25, 60)) + imputer.statistics_, columns=feats)
    X.iloc[0, 3] = np.nan  # imputer must handle missing values
    prob = model.predict_proba(imputer.transform(X))[:, 1]
    assert prob.shape == (25,) and np.all((prob >= 0) & (prob <= 1))


@needs_models
def test_v1_artifacts_load():
    feats = joblib.load(MODELS_DIR / "features_v1.joblib")
    assert len(feats) == 20 and "month" in feats
    assert joblib.load(MODELS_DIR / "waterguard_rf_v1.joblib").n_features_in_ == 20
