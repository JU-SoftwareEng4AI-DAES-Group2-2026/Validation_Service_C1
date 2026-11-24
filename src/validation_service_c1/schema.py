import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SensorRecord(BaseModel):
    """Define and validate one machine's record at one point in time."""

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )

    # Parse the supplied timestamp into a Python datetime.
    # Example JSON value: "2012-01-14T00:00:00"
    timestamp: datetime

    # Identify the machine that produced this record.
    # Require at least one character, so an empty string is rejected.
    machine_id: str = Field(min_length=1)

    # Accept only these operating modes, using the issue's exact spelling.
    operating_mode: Literal[
        "HIGH_LOAD",
        "NORMAL",
        "IDLE",
        "OVERLOAD",
    ]

    temperature: float | None
    vibration: float | None
    pressure: float | None
    rpm: float | None
    power: float | None

    @field_validator("operating_mode", mode="before")
    @classmethod
    def _normalise_operating_mode(cls, value):
        """
        Accept common spellings and convert them to the dataset's form:
            " high_load " / "high load" / "High-Load" -> "HIGH_LOAD"
            "normal load"                             -> "NORMAL"  (PDF spelling)
        Non-strings are returned unchanged so pydantic reports the type error.
        """
        if not isinstance(value, str):
            return value
        cleaned = re.sub(r"[\s\-]+", "_", value.strip()).upper()
        return "NORMAL" if cleaned == "NORMAL_LOAD" else cleaned


class SensorBatch(BaseModel):
    """Define a request containing one or more sensor records.Validate every item using SensorRecord.
    Records may come from different machines.Require at least one record, rejecting {"records": []}"""

    model_config = ConfigDict(extra="forbid")
    records: list[SensorRecord] = Field(min_length=1)
