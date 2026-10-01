"""Smoke-test every dashboard page with Streamlit's AppTest (no browser)."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["page_overview", "page_monitor", "page_explain", "page_inspect", "page_timeline", "page_sensors", "page_performance", "page_method"]


def test_app_with_navigation_runs():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("page", PAGES)
def test_each_page_renders(page, monkeypatch):
    monkeypatch.setenv("WATERGUARD_TEST_PAGE", page)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception, at.exception
    assert len(at.main.children) > 0


def test_explainer_any_timestamp_mode(monkeypatch):
    monkeypatch.setenv("WATERGUARD_TEST_PAGE", "page_explain")
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.button_group[0].set_value("Any timestamp").run()
    assert not at.exception, at.exception


def test_monitor_threshold_change(monkeypatch):
    monkeypatch.setenv("WATERGUARD_TEST_PAGE", "page_monitor")
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.slider[0].set_value(0.5).run()
    assert not at.exception, at.exception


def test_inspection_any_timestamp_with_truth_toggle(monkeypatch):
    monkeypatch.setenv("WATERGUARD_TEST_PAGE", "page_inspect")
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.button_group[0].set_value("Any timestamp").run()  # default date 10 Oct 2018 lies inside the p158 episode
    at.toggle[0].set_value(True).run()
    assert not at.exception, at.exception
