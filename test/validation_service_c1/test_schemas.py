from datetime import datetime

import pytest
from pydantic import ValidationError

from validation_service_c1.schema import SensorBatch, SensorRecord


@pytest.fixture
def valid_record():
    """Provide a fresh, valid dictionary for each test."""
    return {
        "timestamp": "2012-01-14T00:00:00",
        "machine_id": "M001",
        "operating_mode": "HIGH_LOAD",
        "temperature": 46.5,
        "vibration": 1.6,
        "pressure": 98.8,
        "rpm": 2188.4,
        "power": 316.2,
    }


# ---------------------------------------------------------------------------
# Valid input
# ---------------------------------------------------------------------------
def test_valid_record_is_accepted(valid_record):
    """A complete record should validate and parse its timestamp."""
    record = SensorRecord.model_validate(valid_record)

    assert record.machine_id == "M001"
    assert record.temperature == 46.5
    assert record.timestamp == datetime(2012, 1, 14, 0, 0)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" high_load ", "HIGH_LOAD"),
        ("high load", "HIGH_LOAD"),
        ("High-Load", "HIGH_LOAD"),
        ("normal load", "NORMAL"),
        ("normal", "NORMAL"),
        ("idle", "IDLE"),
        ("OVERLOAD", "OVERLOAD"),
    ],
)
def test_operating_mode_is_normalised(valid_record, raw, expected):
    valid_record["operating_mode"] = raw

    record = SensorRecord.model_validate(valid_record)

    assert record.operating_mode == expected


@pytest.mark.parametrize(
    "sensor", ["temperature", "vibration", "pressure", "rpm", "power"]
)
def test_missing_sensor_value_is_accepted(valid_record, sensor):
    """Missing measurements must be available for later imputation."""
    valid_record[sensor] = None

    record = SensorRecord.model_validate(valid_record)

    assert getattr(record, sensor) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", 1052.5),  # sensor spike
        ("vibration", -0.12),  # negative, exists in the real data
    ],
)
def test_out_of_range_values_are_accepted(valid_record, field, value):
    """Only types are validated; suspicious values must reach the ML model."""
    valid_record[field] = value

    record = SensorRecord.model_validate(valid_record)

    assert getattr(record, field) == value


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------
def test_omitted_required_field_is_rejected(valid_record):
    """A nullable value is allowed, but its field must still be present."""
    del valid_record["temperature"]

    with pytest.raises(ValidationError) as exc:
        SensorRecord.model_validate(valid_record)

    assert exc.value.errors()[0]["loc"] == ("temperature",)
    assert exc.value.errors()[0]["type"] == "missing"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("timestamp", "not-a-date"),
        ("machine_id", int),
        ("operating_mode", "UNKNOWN"),
        ("operating_mode", 5),
        ("temperature", "hot"),
        ("temperature", float("nan")),
        ("temperature", float("inf")),
        ("temperature", float("-inf")),
    ],
)
def test_invalid_field_is_rejected(valid_record, field, value):
    """Reject wrong types / unknown categories and name the field responsible."""
    valid_record[field] = value

    with pytest.raises(ValidationError) as exc:
        SensorRecord.model_validate(valid_record)

    assert exc.value.errors()[0]["loc"] == (field,)


def test_unexpected_record_field_is_rejected(valid_record):
    """Unexpected fields should not be silently ignored."""
    valid_record["temprature"] = 46.5

    with pytest.raises(ValidationError) as exc:
        SensorRecord.model_validate(valid_record)

    assert exc.value.errors()[0]["loc"] == ("temprature",)
    assert exc.value.errors()[0]["type"] == "extra_forbidden"


def test_missing_machine_id_is_rejected(valid_record):
    """Every record must say which machine it came from."""
    del valid_record["machine_id"]

    with pytest.raises(ValidationError) as exc:
        SensorRecord.model_validate(valid_record)

    assert exc.value.errors()[0]["loc"] == ("machine_id",)
    assert exc.value.errors()[0]["type"] == "missing"


def test_null_machine_id_is_rejected(valid_record):
    """null is not a machine id either (only sensor readings may be null)."""
    valid_record["machine_id"] = None

    with pytest.raises(ValidationError) as exc:
        SensorRecord.model_validate(valid_record)

    assert exc.value.errors()[0]["loc"] == ("machine_id",)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------
def test_batch_accepts_multiple_machines(valid_record):
    """A batch may contain records from different machines."""
    second_record = valid_record.copy()
    second_record["machine_id"] = "M002"

    batch = SensorBatch.model_validate({"records": [valid_record, second_record]})

    assert [record.machine_id for record in batch.records] == ["M001", "M002"]


def test_empty_batch_is_rejected():
    """A request must contain at least one record."""
    with pytest.raises(ValidationError) as exc:
        SensorBatch.model_validate({"records": []})

    assert exc.value.errors()[0]["loc"] == ("records",)


def test_batch_rejects_an_invalid_record(valid_record):
    """Validation must check every record, including later items."""
    invalid_record = valid_record.copy()
    invalid_record["temperature"] = "hot"

    with pytest.raises(ValidationError) as exc:
        SensorBatch.model_validate({"records": [valid_record, invalid_record]})

    # Index 1 identifies the second record in the batch.
    assert exc.value.errors()[0]["loc"] == ("records", 1, "temperature")


def test_missing_records_field_is_rejected():
    """The request must include the "records" key."""
    with pytest.raises(ValidationError) as exc:
        SensorBatch.model_validate({})

    assert exc.value.errors()[0]["loc"] == ("records",)
    assert exc.value.errors()[0]["type"] == "missing"


def test_unexpected_batch_field_is_rejected(valid_record):
    """Only "records" is allowed at the top level of the request."""
    with pytest.raises(ValidationError) as exc:
        SensorBatch.model_validate({"records": [valid_record], "unexpected": "value"})

    assert exc.value.errors()[0]["loc"] == ("unexpected",)
    assert exc.value.errors()[0]["type"] == "extra_forbidden"
