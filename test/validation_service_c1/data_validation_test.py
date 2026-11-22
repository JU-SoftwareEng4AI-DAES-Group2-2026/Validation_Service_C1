"""M1 contract tests using small observations and temporary CSV/Parquet files."""

import csv
import io
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from validation_service_c1.data_validation import (
    EVALUATION_ONLY_COLUMNS,
    EXCLUDED_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    ML_FEATURE_COLUMNS,
    TRAINING_TARGET_COLUMNS,
    SensorObservation,
    read_sensor_csv,
    read_sensor_parquet,
    select_ml_features,
)


@pytest.fixture
def observation() -> dict:
    return {
        "timestamp": "2012-01-14 00:00:00",
        "machine_id": "M001",
        "operating_mode": "NORMAL",
        "temperature": "46.1",
        "vibration": "1.6",
        "pressure": "98.8",
        "rpm": "2188.4",
        "power": "316.2",
    }


def csv_text(rows: list[list]) -> str:
    """Use CSV quoting rules, including embedded commas and outer wrapping."""
    output = io.StringIO(newline="")
    csv.writer(output).writerows(rows)
    return output.getvalue()


def test_runtime_conversion_without_training_labels(observation: dict) -> None:
    result = SensorObservation.model_validate(observation)
    assert result.timestamp == datetime(2012, 1, 14)
    assert result.machine_id == "M001"
    assert result.operating_mode == "NORMAL"
    for name in ML_FEATURE_COLUMNS[1:]:
        assert isinstance(getattr(result, name), float)
        assert getattr(result, name) == float(observation[name])
    assert set(result.model_dump()) == set(observation)


@pytest.mark.parametrize("mode", ["NORMAL", "HIGH_LOAD", "IDLE", "OVERLOAD"])
def test_operating_modes(observation: dict, mode: str) -> None:
    assert (
        SensorObservation(**{**observation, "operating_mode": mode}).operating_mode
        == mode
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timestamp", "not a timestamp"),
        ("timestamp", "2012-02-30 00:00:00"),
        ("timestamp", 123),
        ("timestamp", None),
        ("machine_id", "M000"),
        ("machine_id", "M1"),
        ("machine_id", "m001"),
        ("machine_id", "M001\n"),
        ("machine_id", 1),
        ("operating_mode", "normal"),
        ("operating_mode", "UNKNOWN"),
        ("operating_mode", None),
    ],
)
def test_invalid_metadata(observation: dict, field: str, value: object) -> None:
    with pytest.raises(ValidationError) as error:
        SensorObservation(**{**observation, field: value})
    assert error.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize("field", ML_FEATURE_COLUMNS[1:])
@pytest.mark.parametrize("value", ["broken", True, [], float("inf"), "-inf"])
def test_invalid_measurements(observation: dict, field: str, value: object) -> None:
    with pytest.raises(ValidationError) as error:
        SensorObservation(**{**observation, field: value})
    assert error.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize("field", tuple(SensorObservation.model_fields))
def test_required_fields(observation: dict, field: str) -> None:
    observation.pop(field)
    with pytest.raises(ValidationError) as error:
        SensorObservation(**observation)
    assert error.value.errors()[0]["type"] == "missing"
    assert error.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize("field", ML_FEATURE_COLUMNS[1:])
@pytest.mark.parametrize("value", [None, "", "  ", float("nan"), "NaN"])
def test_missing_measurements(observation: dict, field: str, value: object) -> None:
    result = SensorObservation(**{**observation, field: value})
    assert getattr(result, field) is None


@pytest.mark.parametrize("value", [-1e12, 0, 1e12])
def test_unusual_finite_values_are_preserved(observation: dict, value: float) -> None:
    assert (
        SensorObservation(**{**observation, "temperature": value}).temperature == value
    )


def test_csv_mixed_quoting_preserves_all_data(
    tmp_path: Path, observation: dict
) -> None:
    header = [*observation, "important_features", "anomaly"]
    row = [*observation.values(), "vibration,rpm", "False"]
    missing_row = row.copy()
    missing_row[header.index("temperature")] = ""
    text = csv_text([header, row]) + csv_text(
        [[csv_text([missing_row]).rstrip("\r\n")]]
    )
    path = tmp_path / "mixed.csv"
    path.write_text(text, encoding="utf-8", newline="")
    original = path.read_bytes()
    frame = read_sensor_csv(path)
    assert frame.shape == (2, len(header))
    assert frame["important_features"].to_list() == ["vibration,rpm"] * 2
    assert frame["temperature"].to_list() == [46.1, None]
    assert frame["timestamp"].to_list() == [datetime(2012, 1, 14)] * 2
    assert frame["anomaly"].to_list() == ["False", "False"]
    assert path.read_bytes() == original


def test_header_only_csv_has_typed_columns(tmp_path: Path, observation: dict) -> None:
    path = tmp_path / "empty.csv"
    path.write_text(csv_text([list(observation)]), encoding="utf-8", newline="")
    frame = read_sensor_csv(path)
    assert frame.height == 0
    assert frame.schema["temperature"] == pl.Float64
    assert frame.schema["timestamp"] == pl.Datetime("us")


@pytest.mark.parametrize("kind", ["short", "long", "invalid", "blank", "quoting"])
def test_csv_invalid_rows_fail_with_line_number(
    tmp_path: Path, observation: dict, kind: str
) -> None:
    row = list(observation.values())
    if kind == "short":
        row.pop()
    elif kind == "long":
        row.append("extra")
    elif kind == "invalid":
        row[0] = "invalid timestamp"
    text = csv_text([list(observation)])
    text += {"blank": "\n", "quoting": '"unterminated'}.get(kind, csv_text([row]))
    path = tmp_path / "invalid.csv"
    path.write_text(text, encoding="utf-8", newline="")
    with pytest.raises(ValueError, match="CSV line 2"):
        read_sensor_csv(path)


@pytest.mark.parametrize(
    "header", [[], [""], ["timestamp", "timestamp"], ["timestamp"]]
)
def test_invalid_csv_headers(tmp_path: Path, header: list[str]) -> None:
    path = tmp_path / "header.csv"
    path.write_text(csv_text([header]), encoding="utf-8", newline="")
    with pytest.raises(ValueError, match="CSV"):
        read_sensor_csv(path)


def test_feature_selection_prevents_leakage(observation: dict) -> None:
    labels = (
        TRAINING_TARGET_COLUMNS + EVALUATION_ONLY_COLUMNS + EXCLUDED_FEATURE_COLUMNS
    )
    record = {**observation, **dict.fromkeys(labels, "ground truth"), "future_label": 1}
    frame = pl.DataFrame([record])
    features = select_ml_features(frame)
    assert features.columns == list(ML_FEATURE_COLUMNS)
    assert features.to_dicts() == [
        {name: observation[name] for name in ML_FEATURE_COLUMNS}
    ]
    assert all(name in frame.columns for name in METADATA_COLUMNS + labels)
    runtime = SensorObservation.model_validate(record)
    assert set(runtime.model_dump()).isdisjoint(labels)


def test_feature_selection_requires_every_feature(observation: dict) -> None:
    frame = pl.DataFrame([observation]).drop("power")
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        select_ml_features(frame)


@pytest.mark.parametrize("typed_timestamp", [False, True])
def test_parquet_validation_preserves_rows_and_labels(
    tmp_path: Path, observation: dict, typed_timestamp: bool
) -> None:
    path = tmp_path / "sensors.parquet"
    row = {**observation, "anomaly": False, "important_features": "vibration,rpm"}
    if typed_timestamp:
        row["timestamp"] = datetime(2012, 1, 14)
    data = pl.DataFrame([row, {**row, "temperature": None}])
    data.write_parquet(path)
    original = path.read_bytes()

    result = read_sensor_parquet(path)

    assert result.shape == data.shape
    assert result.columns == data.columns
    assert result["timestamp"].to_list() == [datetime(2012, 1, 14)] * 2
    assert result["temperature"].to_list() == [46.1, None]
    assert result["rpm"].dtype == pl.Float64
    assert result["rpm"].to_list() == [2188.4] * 2
    assert result["anomaly"].dtype == pl.Boolean
    assert result["anomaly"].to_list() == [False, False]
    assert result["important_features"].to_list() == ["vibration,rpm"] * 2
    assert select_ml_features(result).columns == list(ML_FEATURE_COLUMNS)
    assert path.read_bytes() == original


def test_parquet_invalid_observation_reports_row(
    tmp_path: Path, observation: dict
) -> None:
    path = tmp_path / "invalid.parquet"
    pl.DataFrame([observation, {**observation, "rpm": "broken"}]).write_parquet(path)
    with pytest.raises(
        ValueError,
        match="Parquet row 2:.*",
    ):
        read_sensor_parquet(path)


def test_parquet_requires_columns_even_when_empty(
    tmp_path: Path, observation: dict
) -> None:
    path = tmp_path / "missing.parquet"
    pl.DataFrame([observation]).drop("power").head(0).write_parquet(path)
    with pytest.raises(ValueError, match="missing required columns:.*power"):
        read_sensor_parquet(path)


def test_empty_parquet_has_typed_runtime_columns(
    tmp_path: Path, observation: dict
) -> None:
    path = tmp_path / "empty.parquet"
    pl.DataFrame([{**observation, "anomaly": False}]).head(0).write_parquet(path)
    result = read_sensor_parquet(path)
    assert result.height == 0
    assert result.schema["timestamp"] == pl.Datetime("us")
    assert result.schema["temperature"] == pl.Float64
    assert result.schema["anomaly"] == pl.Boolean
