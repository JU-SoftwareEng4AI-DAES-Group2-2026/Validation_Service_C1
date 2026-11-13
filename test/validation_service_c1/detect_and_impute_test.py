"""Tests for the detect-and-impute pipeline, using small in-memory data."""

from datetime import datetime

import polars as pl
import pytest

from validation_service_c1 import detect_and_impute
from validation_service_c1.detect_and_impute import (
    SENSOR_DATA,
    dai,
    detect,
    flag_missing,
    impute,
    load_data,
)


@pytest.fixture
def df() -> pl.DataFrame:
    """Four rows where row 3 is missing temperature and row 1 is missing rpm.

    Row 3's other sensors sit between rows 0 and 1, so those are its two
    nearest neighbours.
    """
    return pl.DataFrame(
        {
            "machine_id": ["a", "b", "c", "d"],
            "temperature": [10.0, 20.0, 100.0, None],
            "vibration": [1.0, 2.0, 10.0, 1.5],
            "pressure": [1.0, 2.0, 10.0, 1.5],
            "rpm": [1.0, None, 10.0, 1.5],
            "power": [1.0, 2.0, 10.0, 1.5],
        }
    )


def test_detect_counts_nulls_per_sensor(df):
    """Each sensor's null count is reported; other columns are left out."""
    counts = detect(df)

    assert counts.columns == SENSOR_DATA
    assert counts.row(0, named=True) == {
        "temperature": 1,
        "vibration": 0,
        "pressure": 0,
        "rpm": 1,
        "power": 0,
    }


def test_flag_missing_marks_only_null_values(df):
    """Each sensor gets a flag column that is True exactly where it is null."""
    out = flag_missing(df)

    assert out["temperature_imputed"].to_list() == [False, False, False, True]
    assert out["rpm_imputed"].to_list() == [False, True, False, False]
    assert not out["power_imputed"].any()
    assert out.select(df.columns).equals(df)  # original columns unchanged


def test_impute_uses_mean_of_two_nearest_rows(df):
    """Row 3's temperature becomes the mean of rows 0 and 1 (10 and 20)."""
    out = impute(df)

    assert out["temperature"][3] == pytest.approx(15.0)


def test_impute_keeps_known_values_and_other_columns(df):
    """Only nulls change; non-sensor columns pass through untouched."""
    out = impute(df)

    assert out.select(SENSOR_DATA).null_count().row(0) == (0,) * len(SENSOR_DATA)
    assert out["temperature"][:3].to_list() == [10.0, 20.0, 100.0]
    assert out["machine_id"].equals(df["machine_id"])


def test_dai_flags_and_fills_missing_values(df):
    """The pipeline flags the original nulls and leaves none behind."""
    out = dai(df)

    assert out["temperature_imputed"].to_list() == [False, False, False, True]
    assert out["rpm_imputed"].to_list() == [False, True, False, False]
    assert out.select(SENSOR_DATA).null_count().sum_horizontal().item() == 0


def test_load_data_parses_csv_stored_in_one_column(monkeypatch):
    """The header-as-column-name CSV text is rebuilt and parsed, dates included."""
    sheet = pl.DataFrame(
        {
            "timestamp,temperature,rpm": [
                "2024-01-01 00:00:00,20.5,",
                "2024-01-01 01:00:00,,1500",
            ]
        }
    )
    monkeypatch.setattr(detect_and_impute.pl, "read_excel", lambda _path: sheet)

    out = load_data("ignored.xlsx")

    assert out.columns == ["timestamp", "temperature", "rpm"]
    assert out["timestamp"].to_list() == [
        datetime(2024, 1, 1, 0),
        datetime(2024, 1, 1, 1),
    ]
    assert out["temperature"].to_list() == [20.5, None]
    assert out["rpm"].to_list() == [None, 1500]
