"""Tests for the detect-and-impute pipeline, using small in-memory data."""

from datetime import datetime

import polars as pl
import pytest

from validation_service_c1.detect_and_impute import (
    SENSOR_DATA,
    detect,
    detect_and_impute,
    flag_missing,
    impute,
    load_data,
)


@pytest.fixture
def df() -> pl.DataFrame:
    """A chunk from two machines with one missing temperature and rpm.

    Row 3 (machine a, NORMAL) is missing temperature; row 4 (machine a, IDLE)
    is missing rpm and is the only IDLE row for that machine.
    """
    return pl.DataFrame(
        {
            "machine_id": ["a", "a", "a", "a", "a", "b"],
            "operating_mode": [
                "NORMAL",
                "NORMAL",
                "HIGH_LOAD",
                "NORMAL",
                "IDLE",
                "NORMAL",
            ],
            "temperature": [10.0, 20.0, 100.0, None, 40.0, 500.0],
            "vibration": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "pressure": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "rpm": [100.0, 200.0, 300.0, 400.0, None, 900.0],
            "power": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
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

    assert out["temperature_imputed"].to_list() == [
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert out["rpm_imputed"].to_list() == [False, False, False, False, True, False]
    assert not out["power_imputed"].any()
    assert out.select(df.columns).equals(df)  # original columns unchanged


def test_impute_uses_median_of_same_machine_and_mode(df):
    """Row 3 gets the median of machine a's other NORMAL rows (10 and 20).

    Machine a's HIGH_LOAD reading and machine b's reading are ignored.
    """
    out = impute(df)

    assert out["temperature"][3] == pytest.approx(15.0)


def test_impute_falls_back_to_machine_median_when_mode_has_no_readings(df):
    """Row 4 is machine a's only IDLE row, so it gets the machine's rpm median."""
    out = impute(df)

    assert out["rpm"][4] == pytest.approx(250.0)  # median of 100, 200, 300, 400


def test_impute_leaves_null_when_machine_has_no_readings():
    """With no readings for a sensor on a machine, there is nothing to fill from."""
    chunk = pl.DataFrame(
        {
            "machine_id": ["a", "b"],
            "operating_mode": ["NORMAL", "NORMAL"],
            **{c: [1.0, 2.0] for c in SENSOR_DATA},
        }
    ).with_columns(temperature=pl.Series([None, 2.0]))

    out = impute(chunk)

    assert out["temperature"].to_list() == [None, 2.0]


def test_impute_keeps_known_values_and_other_columns(df):
    """Only nulls change; non-sensor columns pass through untouched."""
    out = impute(df)

    assert out.select(SENSOR_DATA).null_count().row(0) == (0,) * len(SENSOR_DATA)
    assert out["temperature"][:3].to_list() == [10.0, 20.0, 100.0]
    assert out.select("machine_id", "operating_mode").equals(
        df.select("machine_id", "operating_mode")
    )


def test_dai_flags_and_fills_missing_values(df):
    """The pipeline flags the original nulls and leaves none behind."""
    out = detect_and_impute(df)

    assert out["temperature_imputed"].to_list() == [
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert out["rpm_imputed"].to_list() == [False, False, False, False, True, False]
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
    monkeypatch.setattr(pl, "read_excel", lambda _path: sheet)

    out = load_data("ignored.xlsx")

    assert out.columns == ["timestamp", "temperature", "rpm"]
    assert out["timestamp"].to_list() == [
        datetime(2024, 1, 1, 0),
        datetime(2024, 1, 1, 1),
    ]
    assert out["temperature"].to_list() == [20.5, None]
    assert out["rpm"].to_list() == [None, 1500]
