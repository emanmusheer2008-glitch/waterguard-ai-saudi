"""Dataset parsing and integrity."""
import io

import pandas as pd

from conftest import needs_leakage, needs_scada
from waterguard.data import (EXPECTED_LEAK_LOCATIONS, EXPECTED_SENSOR_COUNTS, leak_columns,
                             load_leakages, load_scada, validate_scada)
from waterguard.target import add_target

SAMPLE = "Timestamp;p31;p158\n2018-01-01 00:00;0;1,5\n2018-01-01 00:05;2,25;0\n"


def test_leakage_parser_handles_semicolon_and_decimal_comma(tmp_path):
    f = tmp_path / "leaks.csv"
    f.write_text(SAMPLE)
    df = load_leakages(f)
    assert list(df.columns) == ["Timestamp", "p31", "p158"]
    assert df["p158"].iloc[0] == 1.5 and df["p31"].iloc[1] == 2.25
    assert pd.api.types.is_datetime64_any_dtype(df["Timestamp"])


def test_default_csv_parsing_would_be_wrong():
    """Documents why sep=';' / decimal=',' is required."""
    naive = pd.read_csv(io.StringIO(SAMPLE))
    assert naive.shape[1] == 1


def test_target_uses_total_leak_threshold(tmp_path):
    f = tmp_path / "leaks.csv"
    f.write_text("Timestamp;a;b\n2018-01-01 00:00;20;19,9\n2018-01-01 00:05;20;20\n")
    t = add_target(load_leakages(f), threshold=40)
    assert t["total_leak"].round(2).tolist() == [39.9, 40.0]
    assert t["target"].tolist() == [0, 1]


@needs_leakage
def test_real_leakage_file_structure():
    df = load_leakages()
    assert len(df) == 105_120
    assert leak_columns(df) == EXPECTED_LEAK_LOCATIONS
    assert df.isna().sum().sum() == 0
    assert not df["Timestamp"].duplicated().any()
    share_any = (df[leak_columns(df)] > 0).any(axis=1).mean()
    assert abs(share_any - 0.978) < 0.001


@needs_scada
def test_real_scada_columns_and_alignment():
    from waterguard.config import DATA_DIR
    scada = load_scada(cache_dir=DATA_DIR / "cache")
    report = validate_scada(scada)
    for key, n in EXPECTED_SENSOR_COUNTS.items():
        assert report[key]["sensors"] == n
        assert report[key]["rows"] == 105_120
        assert report[key]["duplicates"] == 0
        assert report[key]["missing_values"] == 0
        assert report[key]["irregular_steps"] == 0
        assert report[key]["aligned_with_pressures"]
