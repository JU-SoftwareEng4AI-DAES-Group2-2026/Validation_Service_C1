"""Tests for frozen-sensor and spike detection, using small in-memory data."""

from datetime import datetime, timedelta

import polars as pl
import pytest

from validation_service_c1.corrupted_measurements import (
    detect_corrupted,
    detect_remove_and_impute,
    flag_frozen,
    flag_spikes,
    remove_corrupted,
)
from validation_service_c1.detect_and_impute import SENSOR_DATA

START = datetime(2024, 1, 1)


def chunk(machine_ids, **sensors) -> pl.DataFrame:
    """Hourly NORMAL rows; sensors not given get distinct, ordinary values."""
    n = len(machine_ids)
    defaults = {
        c: [50.0 + i + 0.1 * k for i in range(n)] for k, c in enumerate(SENSOR_DATA)
    }
    return pl.DataFrame(
        {
            "timestamp": [START + timedelta(hours=i) for i in range(n)],
            "machine_id": machine_ids,
            "operating_mode": ["NORMAL"] * n,
            **(defaults | sensors),
        }
    )


def test_flag_frozen_marks_every_value_in_a_run_of_three():
    """Three equal values in a row are all frozen; a pair is not."""
    df = chunk(["a"] * 7, rpm=[1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0])

    out = flag_frozen(df)

    assert out["rpm_frozen"].to_list() == [
        False,
        False,
        False,
        True,
        True,
        True,
        False,
    ]


def test_flag_frozen_respects_min_run():
    """With min_run=5, a run of four is not frozen."""
    df = chunk(["a"] * 4, rpm=[7.0] * 4)

    assert not flag_frozen(df, min_run=5)["rpm_frozen"].any()
    assert flag_frozen(df, min_run=4)["rpm_frozen"].all()


def test_flag_frozen_counts_runs_per_machine_in_timestamp_order():
    """Interleaved machines and shuffled rows do not break or join runs."""
    df = chunk(["a", "b", "a", "b", "a", "b"], rpm=[7.0, 7.0, 7.0, 8.0, 7.0, 9.0])

    out = flag_frozen(df.reverse())

    frozen = dict(zip(out["timestamp"], out["rpm_frozen"]))
    assert [frozen[t] for t in df["timestamp"]] == [
        True,
        False,
        True,
        False,
        True,
        False,
    ]


def test_flag_frozen_ignores_nulls():
    """A run of missing readings is missing, not frozen."""
    df = chunk(["a"] * 3, power=[None, None, None])

    assert not flag_frozen(df)["power_frozen"].any()


def test_flag_frozen_marks_known_stuck_value():
    """A single vibration reading of exactly 5.0 is frozen; 5.01 is not."""
    df = chunk(["a"] * 3, vibration=[4.0, 5.0, 5.01])

    assert flag_frozen(df)["vibration_frozen"].to_list() == [False, True, False]


def test_flag_frozen_stuck_value_only_applies_to_its_sensor():
    """5.0 is only a stuck value for vibration."""
    df = chunk(["a"] * 3, rpm=[4.0, 5.0, 6.0])

    assert not flag_frozen(df)["rpm_frozen"].any()


def test_flag_spikes_marks_temperature_above_limit():
    """Temperatures above 500 are spikes even next to other high readings."""
    df = chunk(["a"] * 4, temperature=[1050.0, 1060.0, 499.0, None])

    assert flag_spikes(df)["temperature_spike"].to_list() == [
        True,
        True,
        False,
        False,
    ]


def test_flag_spikes_limit_only_applies_to_temperature():
    """Other sensors have no fixed limit: a steady high rpm is fine."""
    df = chunk(["a"] * 3, rpm=[2600.0, 2610.0, 2590.0])

    assert not flag_spikes(df)["rpm_spike"].any()


def test_flag_spikes_marks_jump_against_both_neighbours():
    """1050 is >10x both neighbours; the readings around it are not spikes."""
    df = chunk(["a"] * 4, temperature=[50.0, 1050.0, 51.0, 49.0])

    out = flag_spikes(df)

    assert out["temperature_spike"].to_list() == [False, True, False, False]


def test_flag_spikes_marks_drop_as_well_as_rise():
    """A reading 10x smaller than both neighbours is also a spike."""
    df = chunk(["a"] * 3, pressure=[100.0, 2.0, 100.0])

    assert flag_spikes(df)["pressure_spike"].to_list() == [False, True, False]


def test_flag_spikes_ignores_level_change():
    """A lasting jump differs from only one neighbour, so it is not a spike."""
    df = chunk(["a"] * 4, rpm=[50.0, 50.0, 1000.0, 1000.0])

    assert not flag_spikes(df)["rpm_spike"].any()


def test_flag_spikes_uses_two_readings_on_one_side_at_chunk_edges():
    """Edge spikes are caught; their normal neighbours are not flagged."""
    df = chunk(["a"] * 5, temperature=[1050.0, 50.0, 51.0, 49.0, 1050.0])

    assert flag_spikes(df)["temperature_spike"].to_list() == [
        True,
        False,
        False,
        False,
        True,
    ]


def test_flag_spikes_looks_past_a_missing_neighbour():
    """With the next reading null, the one after it is compared instead."""
    df = chunk(["a"] * 4, temperature=[50.0, 1050.0, None, 51.0])

    assert flag_spikes(df)["temperature_spike"].to_list() == [
        False,
        True,
        False,
        False,
    ]


def test_flag_spikes_needs_two_readings_to_compare():
    """Nulls, and readings with fewer than two others, are never spikes."""
    df = chunk(["a", "a", "b"], pressure=[50.0, 1050.0, 1050.0])

    assert not flag_spikes(df)["pressure_spike"].any()


def test_flag_spikes_respects_factor():
    """60 is 1.2x its neighbours: a spike only with a factor below 1.2."""
    df = chunk(["a"] * 3, power=[50.0, 60.0, 50.0])

    assert not flag_spikes(df)["power_spike"].any()
    assert flag_spikes(df, factor=1.1)["power_spike"][1]


def test_detect_corrupted_adds_both_flag_sets():
    """Every sensor gets a frozen and a spike column; inputs are untouched."""
    df = chunk(["a"] * 3, temperature=[50.0, 1050.0, 51.0])

    out = detect_corrupted(df)

    assert out.select(df.columns).equals(df)
    assert set(out.columns) - set(df.columns) == {
        f"{c}_{kind}" for c in SENSOR_DATA for kind in ("frozen", "spike")
    }
    assert out["temperature_spike"].to_list() == [False, True, False]


def test_remove_corrupted_moves_bad_values_to_original_column():
    """Spiking/frozen values are nulled and kept in <sensor>_original."""
    df = chunk(
        ["a"] * 5,
        temperature=[50.0, 1050.0, 51.0, 49.0, 52.0],
        vibration=[4.0, 5.0, 5.0, 5.0, 4.2],
    )

    out = remove_corrupted(df)

    assert out["temperature"].to_list() == [50.0, None, 51.0, 49.0, 52.0]
    assert out["temperature_original"].to_list() == [None, 1050.0, None, None, None]
    assert out["vibration"].to_list() == [4.0, None, None, None, 4.2]
    assert out["vibration_original"].to_list() == [None, 5.0, 5.0, 5.0, None]


def test_detect_remove_and_impute_reports_each_issue():
    """Corrupted and missing values are imputed and reported in quality_flags."""
    df = chunk(
        ["a"] * 5,
        temperature=[50.0, 1050.0, 51.0, 49.0, 52.0],
        vibration=[4.0, 5.0, 5.0, 5.0, 4.2],
        power=[50.0, 51.0, None, 49.0, 52.0],
    )

    out = detect_remove_and_impute(df)

    assert out.select(SENSOR_DATA).null_count().sum_horizontal().item() == 0
    assert out["temperature"][1] == pytest.approx(50.5)  # median of 49-52
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
            "sensor": "vibration",
            "issue": "sensor_frozen",
            "action": "imputed",
            "original_value": 5.0,
        },
        {
            "sensor": "power",
            "issue": "missing_value",
            "action": "imputed",
            "original_value": None,
        },
    ]
    assert out["quality_flags"][0].to_list() == []
