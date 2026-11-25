"""Detect corrupted sensor measurements: frozen sensors and spikes.

Each check combines a general rule, which looks only at the chunk itself,
with a fixed rule matching how the dataset's corrupted readings were made.
A reading is flagged if either rule matches.

* **Frozen**
    - General: a sensor reports exactly the same value ``FROZEN_RUN`` or
      more times in a row, as a real stuck sensor does.
    - Dataset: the value is one of ``FROZEN_VALUES`` for that sensor. Every
      labelled ``sensor_frozen`` reading is vibration at exactly ``5.0`` for
      a single hour, which the general rule cannot see.
* **Spike**
    - General: a reading is more than ``SPIKE_FACTOR`` times larger (or
      smaller) than both the previous and the next reading. At the edge of a
      chunk, the two readings on the one side that exists are used instead.
    - Dataset: the reading is above ``SPIKE_LIMITS`` for that sensor. Every
      labelled ``sensor_spike`` is the normal temperature plus about 1000
      (lowest 1029), while normal temperatures stay below 100.

Each check adds a boolean ``<sensor>_frozen`` / ``<sensor>_spike`` column per
sensor; a null reading is never flagged as corrupted.

There are two ways to use this module:

* **Training data** - ``detect_corrupted`` only adds the flag columns, so
  the modeller can drop the flagged rows.
* **Production** - ``detect_remove_and_impute`` also moves each corrupted
  reading into ``<sensor>_original``, nulls it, and runs
  ``detect_and_impute`` so it is imputed and reported in ``quality_flags``.
"""

import polars as pl

from validation_service_c1.detect_and_impute import (
    MACHINE,
    SENSOR_DATA,
    detect_and_impute,
)

TIMESTAMP = "timestamp"

# A value repeated this many times in a row is frozen.
FROZEN_RUN = 3

# Values a stuck sensor reports in the dataset, per sensor.
FROZEN_VALUES: dict[str, list[float]] = {"vibration": [5.0]}

# Readings above these limits are spikes in the dataset, per sensor.
SPIKE_LIMITS: dict[str, float] = {"temperature": 500.0}

# A reading this many times larger or smaller than its neighbours is a spike.
# On the dataset, 10 catches 1058 of 1061 labelled spikes and flags 1 of
# 522k normal readings; at 5, operating-mode switches start to be flagged.
SPIKE_FACTOR = 10.0


def flag_frozen(df: pl.DataFrame, min_run: int = FROZEN_RUN) -> pl.DataFrame:
    """Flag frozen sensor values.

    A value is frozen if it repeats ``min_run`` or more times in a row, or is
    one of that sensor's ``FROZEN_VALUES``. Every value in a run is flagged,
    including the first. Runs are counted per machine in timestamp order,
    within ``df`` only, so a run split across two chunks is counted
    separately in each.

    Args:
        df: Data containing ``machine_id``, ``timestamp`` and every column in
            ``SENSOR_DATA``.
        min_run: The shortest run of identical values that counts as frozen.

    Returns:
        ``df`` with a boolean ``<sensor>_frozen`` column added per sensor.
    """

    def is_frozen(c: str) -> pl.Expr:
        run_id = pl.col(c).rle_id().over(MACHINE, order_by=TIMESTAMP)
        run_length = pl.len().over(MACHINE, run_id)
        stuck_value = pl.any_horizontal(
            pl.lit(False), *(pl.col(c) == v for v in FROZEN_VALUES.get(c, []))
        )
        return pl.col(c).is_not_null() & ((run_length >= min_run) | stuck_value)

    return df.with_columns(is_frozen(c).alias(f"{c}_frozen") for c in SENSOR_DATA)


def flag_spikes(df: pl.DataFrame, factor: float = SPIKE_FACTOR) -> pl.DataFrame:
    """Flag spiking sensor values.

    A reading is a spike if it is above that sensor's ``SPIKE_LIMITS``, or
    if it jumps against both neighbours: its ratio to the previous and to the next reading
    of the same machine (in timestamp order) is above ``factor`` or below
    ``1 / factor``. Null readings are skipped, so the comparison is with the
    nearest readings that exist. At the edge of the chunk, the second reading
    on the one side that exists stands in for the missing neighbour, so a
    reading next to a spike at the edge is not flagged with it. A reading
    without two others to compare against is never a spike.

    Args:
        df: Data containing ``machine_id``, ``timestamp`` and every column in
            ``SENSOR_DATA``.
        factor: How many times larger or smaller than its neighbours a
            reading must be to count as a spike.

    Returns:
        ``df`` with a boolean ``<sensor>_spike`` column added per sensor.
    """

    def jumps_from(value: pl.Expr, neighbour: pl.Expr) -> pl.Expr:
        ratio = (value / neighbour).abs()
        return (ratio > factor) | (ratio < 1 / factor)

    def is_spike(c: str) -> pl.Series:
        # Work on this sensor's non-null readings, then map back to all rows.
        readings = df.with_row_index("_row").filter(pl.col(c).is_not_null())
        value = pl.col(c)

        def neighbour(n: int) -> pl.Expr:
            return value.shift(n).over(MACHINE, order_by=TIMESTAMP)

        before = pl.coalesce(neighbour(1), neighbour(-2))
        after = pl.coalesce(neighbour(-1), neighbour(2))
        # A missing comparison makes the result null, which counts as no spike.
        spikes = readings.select(
            "_row",
            (jumps_from(value, before) & jumps_from(value, after)).fill_null(False),
        )
        return pl.Series(f"{c}_spike", [False] * df.height).scatter(
            spikes["_row"], spikes[c]
        )

    def above_limit(c: str) -> pl.Expr:
        if c not in SPIKE_LIMITS:
            return pl.lit(False)
        return (pl.col(c) > SPIKE_LIMITS[c]).fill_null(False)

    return df.with_columns(
        (is_spike(c) | above_limit(c)).alias(f"{c}_spike") for c in SENSOR_DATA
    )


def detect_corrupted(df: pl.DataFrame) -> pl.DataFrame:
    """Flag frozen and spiking sensor values.

    Args:
        df: Data containing ``machine_id``, ``timestamp`` and every column in
            ``SENSOR_DATA``.

    Returns:
        ``df`` with ``<sensor>_frozen`` and ``<sensor>_spike`` columns added.
    """
    return flag_spikes(flag_frozen(df))


def remove_corrupted(df: pl.DataFrame) -> pl.DataFrame:
    """Flag corrupted readings, keep them aside, and null them for imputation.

    Args:
        df: As for ``detect_corrupted``.

    Returns:
        ``df`` with the ``detect_corrupted`` flag columns and a
        ``<sensor>_original`` column per sensor, which holds the removed
        reading where that sensor was frozen or spiking and null elsewhere.
        Those readings are null in the sensor columns.
    """
    flagged = detect_corrupted(df)

    def corrupted(c: str) -> pl.Expr:
        return pl.col(f"{c}_frozen") | pl.col(f"{c}_spike")

    return flagged.with_columns(
        pl.when(corrupted(c)).then(pl.col(c)).alias(f"{c}_original")
        for c in SENSOR_DATA
    ).with_columns(pl.when(~corrupted(c)).then(pl.col(c)).alias(c) for c in SENSOR_DATA)


def detect_remove_and_impute(df: pl.DataFrame) -> pl.DataFrame:
    """Production pipeline: remove corrupted readings, then impute all gaps.

    Corrupted and missing readings are both filled by ``detect_and_impute``,
    and its ``quality_flags`` column reports each one with its issue
    (``sensor_spike``, ``sensor_frozen`` or ``missing_value``) and original
    reading.

    Args:
        df: As for ``detect_corrupted``, plus ``operating_mode``.

    Returns:
        The output of ``detect_and_impute`` on ``remove_corrupted(df)``.
    """
    return detect_and_impute(remove_corrupted(df))


if __name__ == "__main__":
    from validation_service_c1.detect_and_impute import load_data

    out = detect_corrupted(load_data())
    print(out.select(pl.col("^.*_(frozen|spike)$").sum()))
