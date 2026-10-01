"""HTTP API: same service, structured responses and errors."""
import pytest

from conftest import ROOT, needs_models

pytestmark = needs_models
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

SAMPLE = (ROOT / "samples" / "waterguard_sample_ltown2018_shifted_to_2026.csv").read_bytes()
SAMPLE_GT = (ROOT / "samples" / "waterguard_sample_ground_truth_shifted_to_2026.csv").read_bytes()


@pytest.fixture(scope="module")
def client():
    from api.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["model_loaded"]


def test_model_info(client):
    j = client.get("/model-info").json()
    assert j["n_features"] == 60 and j["alert_threshold_display"] == 0.22
    assert j["verified_test_metrics"]["precision"] == 0.9424 and "different_network" in j["wording"]


def test_sample_schema_and_files(client):
    j = client.get("/sample-schema").json()
    assert j["total_sensor_columns"] == 119 and j["column_groups"]["pressures"]["count"] == 33
    r = client.get("/sample-files/template")
    assert r.status_code == 200 and r.content.startswith(b"Timestamp,P_n1")
    assert client.get("/sample-files/unknown").status_code == 404


def test_analyze_response(client):
    r = client.post("/analyze", files={"file": ("sample.csv", SAMPLE, "text/csv")})
    assert r.status_code == 200
    j = r.json()
    assert j["mode"] == "user_analysis" and j["validation"]["status"] == "ok"
    assert j["summary"]["n_alerts"] == 258 and len(j["predictions"]) == 1152
    p = j["predictions"][0]
    assert set(p) == {"timestamp", "risk_probability", "alert", "reduced_context", "imputed_values"}
    assert all(0 <= q["risk_probability"] <= 1 for q in j["predictions"])
    assert len(j["top_alerts"]) == 5 and j["evaluation"] is None


def test_analyze_with_ground_truth_and_options(client):
    r = client.post("/analyze", params={"include_predictions": False, "top_alerts": 0},
                    files={"file": ("s.csv", SAMPLE, "text/csv"), "ground_truth": ("g.csv", SAMPLE_GT, "text/csv")})
    j = r.json()
    assert r.status_code == 200 and j["predictions"] is None and j["top_alerts"] == []
    assert j["evaluation"]["matched_rows"] == 1152


def test_analyze_errors(client):
    r = client.post("/analyze", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "UNSUPPORTED_FORMAT"
    r = client.post("/analyze", files={"file": ("x.csv", b"a,b\n1,2\n", "text/csv")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_FAILED"
    assert r.json()["error"]["validation"]["issues"][0]["code"] == "MISSING_TIMESTAMP"
    r = client.post("/analyze", files={"file": ("x.xlsx", b"garbage", "application/octet-stream")})
    assert r.status_code == 400 and r.json()["error"]["code"] == "MALFORMED_XLSX"
    assert "/home" not in r.text and "Users" not in r.text and "site-packages" not in r.text  # no local paths


def test_explain(client):
    r = client.post("/explain", files={"file": ("s.csv", SAMPLE, "text/csv")}, data={"timestamp": "2026-10-06 12:00"})
    assert r.status_code == 200 and len(r.json()["signals"]) == 18 and r.json()["inspection"]["n_sensors"] == 33
    r = client.post("/explain", files={"file": ("s.csv", SAMPLE, "text/csv")}, data={"timestamp": "2020-01-01 00:00"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "TIMESTAMP_NOT_FOUND"


def test_benchmark_endpoints(client):
    j = client.get("/benchmark").json()
    assert j["metrics"]["tp"] == 3433 and j["metrics"]["recall"] == 0.3817 and len(j["episodes"]) == 2
    t = client.get("/benchmark/timeline", params={"resolution": "1h"}).json()
    assert len(t["timestamp"]) == len(t["risk_probability"]) == 2628
    assert client.get("/benchmark/timeline", params={"resolution": "2d"}).status_code == 422
    e = client.get("/benchmark/explain", params={"timestamp": "2018-10-09 14:30"}).json()
    assert e["inspection"]["ranking"][0]["sensor"] == "n644" and e["ground_truth"]["severe"]
    assert client.get("/benchmark/explain", params={"timestamp": "2017-01-01"}).status_code == 404
    n = client.get("/network").json()
    assert len(n["pressure_sensors"]) == 33 and len(n["links"]["id"]) == 909


def test_analyze_groups(client):
    import pandas as pd, io
    from waterguard import service as svc
    from waterguard.schema import wide_to_scada
    df = pd.read_csv(io.BytesIO(SAMPLE))
    parts = wide_to_scada(df, svc.load_bundle().schema)
    files = {g: (f"2026_SCADA_{g}.csv", d.to_csv(index=False, sep=";", decimal=",").encode(), "text/csv") for g, d in parts.items()}
    r = client.post("/analyze-groups", files=files, params={"include_predictions": False})
    assert r.status_code == 200 and r.json()["summary"]["n_alerts"] == 258
    r = client.post("/analyze-groups", files={g: files[g] for g in ("pressures", "demands", "flows")})
    assert r.status_code == 422  # missing 'levels' field
