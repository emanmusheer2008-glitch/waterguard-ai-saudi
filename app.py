"""WaterGuard AI Saudi — Streamlit application.

Six sections: Overview · Analyze Data · Risk & Alerts · Inspection & Sensors · Model & Research · Data Guide.
UI code lives in ui/; all model logic lives in src/waterguard/service.py (shared with the API in api/).
Run:  streamlit run app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ui import common  # noqa: E402
from ui.pages import alerts, analyze, data_guide, inspect_sensors, overview, research  # noqa: E402
from waterguard.dashboard_data import missing_outputs  # noqa: E402
from waterguard.service import model_available  # noqa: E402

st.set_page_config(page_title="WaterGuard AI Saudi", page_icon=":material/water_drop:", layout="wide")
common.inject_css()

missing = missing_outputs()
if missing or not model_available():
    st.error("Missing files: " + ", ".join(missing + ([] if model_available() else ["models/waterguard_rf_v2.joblib"]))
             + ". See README → Installation.")
    st.stop()

PAGES = {
    "overview": st.Page(overview.render, title="Overview", icon=":material/dashboard:", default=True),
    "analyze": st.Page(analyze.render, title="Analyze Data", icon=":material/upload_file:", url_path="analyze"),
    "alerts": st.Page(alerts.render, title="Risk & Alerts", icon=":material/monitoring:", url_path="alerts"),
    "inspect": st.Page(inspect_sensors.render, title="Inspection & Sensors", icon=":material/travel_explore:", url_path="inspect"),
    "research": st.Page(research.render, title="Model & Research", icon=":material/science:", url_path="research"),
    "guide": st.Page(data_guide.render, title="Data Guide", icon=":material/menu_book:", url_path="guide"),
}
common.PAGES.update(PAGES)
RENDER = {"overview": overview.render, "analyze": analyze.render, "alerts": alerts.render,
          "inspect": inspect_sensors.render, "research": research.render, "guide": data_guide.render}

_test_page = os.environ.get("WATERGUARD_TEST_PAGE")  # used only by tests/test_app.py
if _test_page:
    RENDER[_test_page]()
else:
    st.navigation(list(PAGES.values()), position="top").run()
