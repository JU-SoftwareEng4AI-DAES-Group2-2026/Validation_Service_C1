from pathlib import Path

import polars as pl

from validation_service_c1.detect_and_impute import (
    SENSOR_DATA,
    dai,
    flag_missing,
    load_data,
)

DATA = Path(__file__).parents[2] / "predictive_maintenance_data.xlsx"


def test_dai_flags_and_fills_only_missing_values():
    df = pl.DataFrame({c: [1.0, 2.0, 3.0] for c in SENSOR_DATA}).with_columns(
        temperature=pl.Series([1.0, None, 3.0])
    )
    out = dai(df)

    assert out["temperature_imputed"].to_list() == [False, True, False]
    assert out["temperature"].null_count() == 0
    assert out["temperature"][[0, 2]].to_list() == [1.0, 3.0]  # known values kept


def test_flags_match_labelled_missing_values():
    df = flag_missing(load_data(DATA))

    expected = df["anomaly_type"] == "missing_value"
    assert (df["temperature_imputed"] == expected.fill_null(False)).all()
