"""Smoke-test every section of the Streamlit app with AppTest (no browser), in both data modes."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from conftest import needs_models

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["overview", "analyze", "alerts", "inspect", "research", "guide"]
pytestmark = needs_models


def run(page, monkeypatch, state=None):
    monkeypatch.setenv("WATERGUARD_TEST_PAGE", page)
    at = AppTest.from_file(APP, default_timeout=180)
    for k, v in (state or {}).items():
        at.session_state[k] = v
    return at.run()


def test_app_with_navigation_runs():
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("page", PAGES)
def test_each_section_renders(page, monkeypatch):
    at = run(page, monkeypatch)
    assert not at.exception, at.exception
    assert len(at.main.children) > 0


@pytest.fixture(scope="module")
def analysis():
    from waterguard import service as svc
    root = Path(APP).parent / "samples"
    res = svc.analyze((root / "waterguard_sample_ltown2018_shifted_to_2026.csv").read_bytes(), "sample.csv",
                      ground_truth=((root / "waterguard_sample_ground_truth_shifted_to_2026.csv").read_bytes(), "gt.csv"))
    return {"result": res, "filename": "sample.csv", "hash": "x"}


def test_analyze_flow_with_sample_file(monkeypatch):
    at = run("analyze", monkeypatch)
    next(b for b in at.button if "sample" in b.label.lower()).click().run()
    run_btn = next(b for b in at.button if b.label == "Run analysis")
    assert not run_btn.disabled
    run_btn.click().run()
    assert not at.exception, at.exception
    res = at.session_state["analysis"]["result"]
    assert res.ok and res.summary["n_alerts"] == 258 and res.evaluation is None
    assert at.session_state["mode"] == "Your analysis"


@pytest.mark.parametrize("page", ["alerts", "inspect"])
def test_user_mode_pages(page, monkeypatch, analysis):
    at = run(page, monkeypatch, {"analysis": analysis, "mode": "Your analysis"})
    assert not at.exception, at.exception
    at.button_group[1].set_value(at.button_group[1].options[1]).run()  # second view
    assert not at.exception, at.exception


@pytest.mark.parametrize("page", ["alerts", "inspect"])
def test_user_mode_without_analysis_shows_empty_state(page, monkeypatch):
    at = run(page, monkeypatch, {"mode": "Your analysis"})
    assert not at.exception, at.exception


def test_inspection_benchmark_any_timestamp_with_truth_toggle(monkeypatch):
    at = run("inspect", monkeypatch)
    at.button_group[2].set_value("Any timestamp").run()  # default 10 Oct 2018 is inside the p158 episode
    at.toggle[0].set_value(True).run()
    assert not at.exception, at.exception


def test_alerts_threshold_change_and_timeline(monkeypatch):
    at = run("alerts", monkeypatch)
    at.slider[0].set_value(0.5).run()
    assert not at.exception, at.exception
    at.button_group[1].set_value("Detection timeline").run()
    assert not at.exception, at.exception
