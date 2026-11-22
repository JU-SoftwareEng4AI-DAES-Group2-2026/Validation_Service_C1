"""Validate runtime sensor observations before downstream quality processing."""

import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator

ML_FEATURE_COLUMNS = (
    "operating_mode",
    "temperature",
    "vibration",
    "pressure",
    "rpm",
    "power",
)
METADATA_COLUMNS = ("timestamp", "machine_id")
TRAINING_TARGET_COLUMNS = ("anomaly", "anomaly_type")
EVALUATION_ONLY_COLUMNS = ("concept_drift_active",)
EXCLUDED_FEATURE_COLUMNS = (
    "failure_probability_gt",
    "failure_in_7_days",
    "remaining_useful_life_hours",
    "failure_type",
    "important_features",
    "maintenance_action",
    "priority",
)


class SensorObservation(BaseModel):
    """Convert sensor fields while leaving missing measurements for M2.

    All eight fields must be supplied, but measurements may be missing.
    Dataset labels and other extra columns are excluded from this runtime model.
    """

    model_config = ConfigDict(extra="ignore")

    timestamp: datetime
    machine_id: str = Field(pattern=r"^M(00[1-9]|0[1-9][0-9]|[1-9][0-9]{2})$")
    operating_mode: Literal["NORMAL", "HIGH_LOAD", "IDLE", "OVERLOAD"]
    temperature: float | None
    vibration: float | None
    pressure: float | None
    rpm: float | None
    power: float | None

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, value: object) -> object:
        """Accept datetime objects and strings, avoiding implicit Unix epochs."""
        if not isinstance(value, (str, datetime)):
            # Pydantic wraps ValueError in a field-specific ValidationError.
            raise ValueError(  # noqa: TRY004
                "timestamp must be a datetime or a datetime string"
            )
        return value

    @field_validator(
        "temperature", "vibration", "pressure", "rpm", "power", mode="before"
    )
    @classmethod
    def normalize_measurement(cls, value: object) -> float | None:
        """Convert numeric input and represent null, blank, or NaN as missing."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError("measurement must be numeric or missing")  # noqa: TRY004
        try:
            measurement = float(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError("measurement must be numeric or missing") from exc
        if math.isnan(measurement):
            return None
        if not math.isfinite(measurement):
            raise ValueError("measurement must be finite or missing")
        return measurement


def select_ml_features(data: pl.DataFrame) -> pl.DataFrame:
    """Select only approved features, in a stable order, failing if any are absent.

    Metadata remains available on the original dataframe for future time-series
    feature engineering. New columns are excluded unless explicitly allowlisted.
    """
    return data.select(ML_FEATURE_COLUMNS)


def read_sensor_parquet(path: str | Path) -> pl.DataFrame:
    """Read Parquet with the same runtime validation as CSV.

    Missing measurements remain null for M2. Extra columns retain their native
    Parquet types, including training labels. Invalid observations raise an error
    with a one-based row number; no rows are dropped or source files modified.
    """
    data = pl.read_parquet(path)
    missing = set(SensorObservation.model_fields) - set(data.columns)
    if missing:
        raise ValueError(f"Parquet is missing required columns: {sorted(missing)}")

    records = []
    for row_number, record in enumerate(
        data.select(list(SensorObservation.model_fields)).iter_rows(named=True), start=1
    ):
        try:
            records.append(SensorObservation.model_validate(record).model_dump())
        except ValueError as exc:
            raise ValueError(f"Parquet row {row_number}: {exc}") from exc

    schema = {
        "machine_id": pl.String,
        "operating_mode": pl.String,
        **{name: pl.Float64 for name in ML_FEATURE_COLUMNS[1:]},
    }
    if not records:
        schema["timestamp"] = pl.Datetime("us")
        validated = pl.DataFrame(schema=schema)
    else:
        validated = pl.from_dicts(
            records, infer_schema_length=None, schema_overrides=schema
        )
    return data.with_columns(validated.get_columns())


def read_sensor_csv(path: str | Path) -> pl.DataFrame:
    """Read and validate ordinary or additionally quoted CSV observations.

    Runtime fields are converted through SensorObservation. Other columns,
    including ground-truth labels, retain their original CSV string values.
    Invalid records fail with a line number rather than being silently dropped.
    The source file is never modified.
    """
    records = []
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source, strict=True)
        try:
            header = next(reader, None)
            if not header or any(not name for name in header):
                raise ValueError("CSV must have a nonempty header with named columns")
            if len(header) != len(set(header)):
                raise ValueError("CSV header contains duplicate column names")
            missing = set(SensorObservation.model_fields) - set(header)
            if missing:
                raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
            for row in reader:
                # An outer quoting layer makes an entire record a single field.
                if len(row) == 1:
                    decoded = list(csv.reader([row[0]], strict=True))
                    if len(decoded) != 1:
                        raise ValueError(f"CSV line {reader.line_num}: invalid record")
                    row = decoded[0]
                if len(row) != len(header):
                    raise ValueError(
                        f"CSV line {reader.line_num}: expected {len(header)} fields, "
                        f"received {len(row)}"
                    )
                record = dict(zip(header, row, strict=True))
                try:
                    observation = SensorObservation.model_validate(record)
                except ValueError as exc:
                    raise ValueError(f"CSV line {reader.line_num}: {exc}") from exc
                records.append({**record, **observation.model_dump()})
        except csv.Error as exc:
            raise ValueError(f"CSV line {reader.line_num}: {exc}") from exc

    # Explicit types also handle header-only files and entirely missing sensors.
    runtime_schema = {
        "timestamp": pl.Datetime("us"),
        "machine_id": pl.String,
        "operating_mode": pl.String,
        **{name: pl.Float64 for name in ML_FEATURE_COLUMNS[1:]},
    }
    schema = {name: runtime_schema.get(name, pl.String) for name in header}
    return pl.from_dicts(records, schema=schema)
