"""Tests for frozen-sensor and spike detection, using small in-memory data."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from validation_service_c1.corrupted_measurements import (
    Reference,
    detect_corrupted,
    detect_remove_and_impute,
    fit_reference,
    flag_frozen,
    flag_spikes,
    remove_corrupted,
)
from validation_service_c1.detect_and_impute import SENSOR_DATA

START = datetime(2024, 1, 1)


def chunk(machine_ids, modes, **sensors) -> pl.DataFrame:
    """Hourly rows; sensors not given get distinct, ordinary values."""
    n = len(machine_ids)
    defaults = {
        c: [10.0 + i + 0.1 * k for i in range(n)] for k, c in enumerate(SENSOR_DATA)
    }
    return pl.DataFrame(
        {
            "timestamp": [START + timedelta(hours=i) for i in range(n)],
            "machine_id": machine_ids,
            "operating_mode": modes,
            **(defaults | sensors),
        }
    )


@pytest.fixture
def reference() -> Reference:
    """NORMAL readings centre on 50 with spread 1; vibration sticks at 5.0."""
    return Reference(
        median={c: {"NORMAL": 50.0} for c in SENSOR_DATA},
        scale={c: {"NORMAL": 1.0} for c in SENSOR_DATA},
        stuck_values={c: [5.0] if c == "vibration" else [] for c in SENSOR_DATA},
    )


def test_fit_reference_learns_median_scale_and_stuck_values():
    """Median and MAD are per mode; only a frequently repeated value is stuck."""
    n = 12
    history = pl.DataFrame(
        {
            "operating_mode": ["NORMAL"] * n + ["IDLE"] * 3,
            **{
                c: [float(i) for i in range(n)] + [100.0, 101.0, 102.0]
                for c in SENSOR_DATA
            },
        }
    ).with_columns(vibration=pl.Series([5.0] * n + [1.0, 2.0, 3.0]))

    ref = fit_reference(history)

    assert ref.median["temperature"] == {"NORMAL": 5.5, "IDLE": 101.0}
    assert ref.scale["temperature"]["NORMAL"] == pytest.approx(3.0 * 1.4826)
    assert ref.stuck_values["vibration"] == [5.0]
    assert ref.stuck_values["temperature"] == []


def test_flag_frozen_marks_repeat_of_same_machines_previous_reading(reference):
    """Row 2 repeats machine a's row 0; machine b's equal value is not a repeat."""
    df = chunk(
        ["a", "b", "a", "b"], ["NORMAL"] * 4, temperature=[50.0, 50.0, 50.0, 51.0]
    )

    out = flag_frozen(df, reference)

    assert out["temperature_frozen"].to_list() == [False, False, True, False]


def test_flag_frozen_orders_by_timestamp_not_row_order(reference):
    """Rows arriving out of order are compared with the previous hour."""
    df = chunk(["a", "a", "a"], ["NORMAL"] * 3, rpm=[1.0, 2.0, 1.0]).reverse()

    out = flag_frozen(df, reference)

    assert not out["rpm_frozen"].any()


def test_flag_frozen_marks_known_stuck_value(reference):
    """A single reading of the stuck value is frozen, with no repeat needed."""
    df = chunk(["a", "a"], ["NORMAL"] * 2, vibration=[4.0, 5.0])

    out = flag_frozen(df, reference)

    assert out["vibration_frozen"].to_list() == [False, True]


def test_flag_frozen_ignores_nulls(reference):
    """Two missing readings in a row are missing, not frozen."""
    df = chunk(["a", "a"], ["NORMAL"] * 2, power=[None, None])

    out = flag_frozen(df, reference)

    assert out["power_frozen"].to_list() == [False, False]


def test_flag_spikes_marks_values_beyond_threshold(reference):
    """With spread 1, 59 is within 10 deviations of 50; 1050 and -20 are not."""
    df = chunk(["a"] * 4, ["NORMAL"] * 4, temperature=[50.0, 59.0, 1050.0, -20.0])

    out = flag_spikes(df, reference)

    assert out["temperature_spike"].to_list() == [False, False, True, True]


def test_flag_spikes_skips_nulls_and_unknown_modes(reference):
    """No reference for the mode, or no reading, means no spike flag."""
    df = chunk(["a", "a"], ["OVERLOAD", "NORMAL"], temperature=[1050.0, None])

    out = flag_spikes(df, reference)

    assert out["temperature_spike"].to_list() == [False, False]


def test_flag_spikes_follows_drift_of_whole_chunk(reference):
    """A chunk drifted to 1.35x is normal; a spike within it still stands out."""
    rpm = [67.5, 67.0, 68.0, 67.5, 500.0]  # reference median 50 * 1.35 = 67.5
    df = chunk(["a"] * 5, ["NORMAL"] * 5, rpm=rpm)

    out = flag_spikes(df, reference)

    assert out["rpm_spike"].to_list() == [False, False, False, False, True]


def test_flag_spikes_uses_unscaled_reference_for_tiny_chunks(reference):
    """With too few readings to estimate drift, the reference is not scaled."""
    df = chunk(["a", "a"], ["NORMAL"] * 2, rpm=[67.5, 50.0])

    out = flag_spikes(df, reference)

    assert out["rpm_spike"].to_list() == [True, False]


def test_detect_corrupted_adds_both_flag_sets(reference):
    """Every sensor gets a frozen and a spike column; inputs are untouched."""
    df = chunk(["a", "a"], ["NORMAL"] * 2, temperature=[50.0, 1050.0])

    out = detect_corrupted(df, reference)

    assert out.select(df.columns).equals(df)
    assert set(out.columns) - set(df.columns) == {
        f"{c}_{kind}" for c in SENSOR_DATA for kind in ("frozen", "spike")
    }
    assert out["temperature_spike"].to_list() == [False, True]


def test_remove_corrupted_moves_bad_values_to_original_column(reference):
    """Spiking/frozen values are nulled and kept in <sensor>_original."""
    df = chunk(
        ["a"] * 4,
        ["NORMAL"] * 4,
        temperature=[50.0, 1050.0, 51.0, 49.0],
        vibration=[4.0, 5.0, 4.5, 4.2],
    )

    out = remove_corrupted(df, reference)

    assert out["temperature"].to_list() == [50.0, None, 51.0, 49.0]
    assert out["temperature_original"].to_list() == [None, 1050.0, None, None]
    assert out["vibration"].to_list() == [4.0, None, 4.5, 4.2]
    assert out["vibration_original"].to_list() == [None, 5.0, None, None]
    assert out["temperature_spike"].to_list() == [False, True, False, False]


def test_detect_remove_and_impute_reports_each_issue(reference):
    """Corrupted and missing values are imputed and reported in quality_flags."""
    df = chunk(
        ["a"] * 4,
        ["NORMAL"] * 4,
        temperature=[50.0, 1050.0, 51.0, 49.0],
        vibration=[4.0, 5.0, 4.5, 4.2],
        power=[50.0, 51.0, None, 49.0],
    )

    out = detect_remove_and_impute(df, reference)

    assert out.select(SENSOR_DATA).null_count().sum_horizontal().item() == 0
    assert out["temperature"][1] == pytest.approx(50.0)  # median of 49, 50, 51
    assert out["quality_flags"][1].to_list() == [
        {
            "sensor": "temperature",
            "issue": "sensor_spike",
            "action": "imputed",
            "original_value": 1050.0,
        },
        {
            "sensor": "vibration",
            "issue": "sensor_frozen",
            "action": "imputed",
            "original_value": 5.0,
        },
    ]
    assert out["quality_flags"][2].to_list() == [
        {
            "sensor": "power",
            "issue": "missing_value",
            "action": "imputed",
            "original_value": None,
        }
    ]
    assert out["quality_flags"][0].to_list() == []
