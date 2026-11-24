"""Tests for the F1 chunk endpoint (receive_chunks.py).

These are API-level tests: requests go through the real FastAPI app in-process.
M2's imputation maths is covered by detect_and_impute_test.py, so the tests here
only check what F1 itself does: validation, error responses, and the wiring to M2.
The 422 and 500 tests replace M2 with a stub on purpose, to check that M2 is not
called for invalid chunks and that an M2 crash is reported properly.
"""

import pytest
from fastapi.testclient import TestClient

from validation_service_c1 import receive_chunks
from validation_service_c1.receive_chunks import app

client = TestClient(app)


def make_record(hour=0, machine_id="M001", mode="NORMAL", **overrides):
    """Build one valid raw reading. Pass e.g. temperature=None to override a field."""
    record = {
        "timestamp": f"2014-01-14T{hour:02d}:00:00",
        "machine_id": machine_id,
        "operating_mode": mode,
        "temperature": 70.0,
        "vibration": 0.3,
        "pressure": 5.0,
        "rpm": 1500.0,
        "power": 12.0,
    }
    record.update(overrides)
    return record


def post(records):
    return client.post("/v1/chunks", json={"records": records})


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --------------------------------------------------------------------------
# Success path
# --------------------------------------------------------------------------
def test_valid_chunk_returns_200_and_summary():
    response = post([make_record(0), make_record(1)])
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "success"
    assert body["accepted"] == 2
    assert body["machines"] == ["M001"]
    assert len(body["records"]) == 2
    assert all(v == 0 for v in body["imputed_per_sensor"].values())
    assert all(v == 0 for v in body["still_null"].values())


def test_chunk_with_several_machines_lists_each_once():
    records = [
        make_record(0, machine_id="M002"),
        make_record(0, machine_id="M001"),
        make_record(1, machine_id="M002"),
    ]
    body = post(records).json()

    assert body["accepted"] == 3
    assert body["machines"] == ["M001", "M002"]


def test_missing_value_is_imputed_and_flagged():
    """Smoke test that F1 is wired to M2: a null goes in, a filled value comes out."""
    response = post([make_record(0), make_record(1, temperature=None)])
    body = response.json()

    assert response.status_code == 200
    assert body["imputed_per_sensor"]["temperature"] == 1
    assert body["still_null"]["temperature"] == 0

    first, second = body["records"]
    assert second["temperature"] == 70.0
    assert second["temperature_imputed"] is True
    assert first["temperature_imputed"] is False


def test_sensor_null_for_whole_chunk_stays_null():
    """CHUNK_SCHEMA must stop Polars inferring a Null dtype and crashing M2."""
    records = [make_record(0, temperature=None), make_record(1, temperature=None)]
    response = post(records)
    body = response.json()

    assert response.status_code == 200
    assert body["still_null"]["temperature"] == 2
    assert body["imputed_per_sensor"]["temperature"] == 0  # nothing could be filled
    assert all(r["temperature"] is None for r in body["records"])
    # M2 flagged both temperatures as missing before it tried to fill them.
    assert all(r["temperature_imputed"] is True for r in body["records"])


@pytest.mark.parametrize(
    "raw_mode, expected",
    [
        ("normal load", "NORMAL"),
        (" high_load ", "HIGH_LOAD"),
        ("High-Load", "HIGH_LOAD"),
        ("idle", "IDLE"),
    ],
)
def test_operating_mode_spellings_are_normalised(raw_mode, expected):
    body = post([make_record(0, mode=raw_mode)]).json()
    assert body["records"][0]["operating_mode"] == expected


# --------------------------------------------------------------------------
# Validation failures (422): M2 must never be called
# --------------------------------------------------------------------------
@pytest.fixture
def m2_must_not_run(monkeypatch):
    def boom(df):
        raise AssertionError("M2 was called for an invalid chunk")

    monkeypatch.setattr(receive_chunks, "detect_and_impute", boom)


def test_bad_operating_mode_is_rejected_with_location(m2_must_not_run):
    response = post([make_record(0), make_record(1, mode="banana")])
    body = response.json()

    assert response.status_code == 422
    assert body["status"] == "rejected"
    assert body["stage"] == "validation"
    assert "Nothing was passed to M2" in body["message"]

    error = body["errors"][0]
    assert error["record_index"] == 1
    assert error["field"] == "operating_mode"
    assert error["error_type"] == "literal_error"
    assert "record 1, field 'operating_mode'" in error["problem"]


def test_empty_records_is_rejected(m2_must_not_run):
    response = client.post("/v1/chunks", json={"records": []})
    body = response.json()

    assert response.status_code == 422
    assert body["errors"][0]["error_type"] == "too_short"
    assert body["errors"][0]["record_index"] is None


def test_missing_records_key_is_rejected(m2_must_not_run):
    response = client.post("/v1/chunks", json={})

    assert response.status_code == 422
    assert response.json()["errors"][0]["error_type"] == "missing"


def test_missing_required_field_is_rejected(m2_must_not_run):
    record = make_record(0)
    del record["machine_id"]
    body = post([record]).json()

    assert body["errors"][0]["error_type"] == "missing"
    assert body["errors"][0]["field"] == "machine_id"
    assert body["errors"][0]["record_index"] == 0


def test_extra_field_is_rejected(m2_must_not_run):
    # Label columns belong to the full SensorRecord, not to raw readings.
    body = post([make_record(0, anomaly=False)]).json()

    assert body["errors"][0]["error_type"] == "extra_forbidden"
    assert body["errors"][0]["field"] == "anomaly"


def test_empty_machine_id_is_rejected(m2_must_not_run):
    response = post([make_record(0, machine_id="")])
    assert response.status_code == 422


def test_non_numeric_sensor_is_rejected(m2_must_not_run):
    body = post([make_record(0, temperature="hot")]).json()

    assert body["errors"][0]["field"] == "temperature"
    assert body["errors"][0]["error_type"] == "float_parsing"


def test_bad_timestamp_is_rejected(m2_must_not_run):
    response = post([make_record(0, timestamp="not-a-date")])
    assert response.status_code == 422


def test_nan_is_rejected(m2_must_not_run):
    # NaN is not valid JSON, but Python's parser accepts it; the schema must refuse it.
    raw = (
        '{"records":[{"timestamp":"2014-01-14T00:00:00","machine_id":"M001",'
        '"operating_mode":"NORMAL","temperature":NaN,"vibration":0.3,'
        '"pressure":5,"rpm":1500,"power":12}]}'
    )
    response = client.post(
        "/v1/chunks", content=raw, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_all_problems_are_listed(m2_must_not_run):
    records = [make_record(0, mode="banana"), make_record(1, temperature="hot")]
    body = post(records).json()

    assert len(body["errors"]) == 2
    assert {e["record_index"] for e in body["errors"]} == {0, 1}


# --------------------------------------------------------------------------
# M2 failure (500)
# --------------------------------------------------------------------------
def test_m2_crash_returns_500_with_description(monkeypatch):
    def boom(df):
        raise RuntimeError("imputer exploded")

    monkeypatch.setattr(receive_chunks, "detect_and_impute", boom)
    response = post([make_record(0)])
    detail = response.json()["detail"]

    assert response.status_code == 500
    assert detail["status"] == "failed"
    assert detail["stage"] == "imputation (M2)"
    assert "RuntimeError: imputer exploded" in detail["error"]
