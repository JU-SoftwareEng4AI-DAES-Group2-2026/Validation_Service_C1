"""Detect corrupted sensor measurements: frozen sensors and spikes.

Both checks compare a chunk against a ``Reference`` fitted on historical data
(by default the first two years of the parquet dataset; year 3 is what
``tools/chunk_sender.py`` streams):

* **Frozen** - a sensor reports the same value as that machine's previous
  reading, or a value known to be a stuck output. Real readings are
  continuous, so an exact repeat means the sensor stopped updating.
* **Spike** - a value further than ``SPIKE_THRESHOLD`` robust standard
  deviations from the median for that sensor in the same operating mode.
  Concept drift scales readings by a constant factor (rpm runs 1.35x higher
  in late 2014), so the reference is first scaled by the machine's drift
  factor in the chunk; a spike is a single reading and does not move it.

Each check adds a boolean ``<sensor>_frozen`` / ``<sensor>_spike`` column per
sensor; a null reading is never flagged as corrupted.

There are two ways to use this module:

* **Training data** - ``detect_corrupted`` only adds the flag columns, so
  the modeller can drop the flagged rows.
* **Production** - ``detect_remove_and_impute`` also moves each corrupted
  reading into ``<sensor>_original``, nulls it, and runs
  ``detect_and_impute`` so it is imputed and reported in ``quality_flags``.
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import polars as pl

from validation_service_c1.detect_and_impute import (
    DATA_PATH,
    MACHINE,
    MODE,
    SENSOR_DATA,
    detect_and_impute,
    load_data,
)

TIMESTAMP = "timestamp"

# How many robust standard deviations from the median counts as a spike. With
# 24-hour chunks this catches every labelled spike in the dataset and flags
# about 0.03% of normal readings, mostly while drift is ramping up.
SPIKE_THRESHOLD = 10.0

# A machine needs at least this many readings in a chunk to estimate its
# drift factor; with fewer, the reference is used unscaled.
MIN_DRIFT_ROWS = 3

# An exact value seen at least this often in the history is a stuck output.
STUCK_MIN_COUNT = 10

# Scales the median absolute deviation to a standard deviation for normal data.
MAD_TO_STD = 1.4826


@dataclass(frozen=True)
class Reference:
    """What normal readings look like, fitted from historical data.

    Attributes:
        median: ``median[sensor][operating_mode]``, the typical value.
        scale: ``scale[sensor][operating_mode]``, the robust standard
            deviation (scaled median absolute deviation).
        stuck_values: ``stuck_values[sensor]``, exact values a stuck sensor
            reports.
    """

    median: dict[str, dict[str, float]]
    scale: dict[str, dict[str, float]]
    stuck_values: dict[str, list[float]]


def fit_reference(history: pl.DataFrame) -> Reference:
    """Learn per-mode medians, spreads and stuck values from history.

    Args:
        history: Data containing ``operating_mode`` and every column in
            ``SENSOR_DATA``.

    Returns:
        The fitted ``Reference``.
    """
    history = history.with_columns(pl.col(MODE).cast(pl.String))
    median: dict[str, dict[str, float]] = {}
    scale: dict[str, dict[str, float]] = {}
    stuck_values: dict[str, list[float]] = {}
    for c in SENSOR_DATA:
        stats = history.group_by(MODE).agg(
            med=pl.col(c).median(),
            mad=(pl.col(c) - pl.col(c).median()).abs().median(),
        )
        median[c] = dict(zip(stats[MODE], stats["med"]))
        scale[c] = {m: mad * MAD_TO_STD for m, mad in zip(stats[MODE], stats["mad"])}
        counts = history[c].drop_nulls().value_counts(name="n")
        stuck_values[c] = counts.filter(pl.col("n") >= STUCK_MIN_COUNT)[c].to_list()
    return Reference(median=median, scale=scale, stuck_values=stuck_values)


@cache
def load_reference(path: str | Path = DATA_PATH, years: int = 2) -> Reference:
    """Fit a ``Reference`` on the first ``years`` years of the dataset.

    The result is cached, so the parquet file is read once per process.
    """
    df = load_data(path)
    ts = pl.col(TIMESTAMP)
    return fit_reference(df.filter(ts < ts.min().dt.offset_by(f"{years}y")))


def flag_frozen(df: pl.DataFrame, reference: Reference) -> pl.DataFrame:
    """Flag sensor values that are frozen.

    A value is frozen if it equals the same machine's previous reading
    (ordered by timestamp) or is one of the reference's stuck values.

    Args:
        df: Data containing ``machine_id``, ``timestamp`` and every column in
            ``SENSOR_DATA``.
        reference: The fitted reference.

    Returns:
        ``df`` with a boolean ``<sensor>_frozen`` column added per sensor.
    """
    return df.with_columns(
        (
            (pl.col(c) == pl.col(c).shift(1).over(MACHINE, order_by=TIMESTAMP))
            | pl.any_horizontal(
                pl.lit(False), *(pl.col(c) == v for v in reference.stuck_values[c])
            )
        )
        .fill_null(False)
        .alias(f"{c}_frozen")
        for c in SENSOR_DATA
    )


def flag_spikes(df: pl.DataFrame, reference: Reference) -> pl.DataFrame:
    """Flag sensor values that are unreasonably far from normal.

    A value is a spike if it lies more than ``SPIKE_THRESHOLD`` robust
    standard deviations from the reference median for its operating mode,
    after both are scaled by the machine's drift factor: the median of
    ``value / reference median`` over that machine's rows in ``df``. Rows in
    an operating mode the reference has not seen are not flagged.

    Args:
        df: Data containing ``operating_mode`` and every column in
            ``SENSOR_DATA``.
        reference: The fitted reference.

    Returns:
        ``df`` with a boolean ``<sensor>_spike`` column added per sensor.
    """
    mode = pl.col(MODE).cast(pl.String)

    def lookup(table: dict[str, float]) -> pl.Expr:
        return mode.replace_strict(table, default=None, return_dtype=pl.Float64)

    def is_spike(c: str) -> pl.Expr:
        median = lookup(reference.median[c])
        ratio = pl.col(c) / median
        drift = (
            pl.when(ratio.count().over(MACHINE) >= MIN_DRIFT_ROWS)
            .then(ratio.median().over(MACHINE))
            .otherwise(1.0)
        )
        return (pl.col(c) - drift * median).abs() > (
            SPIKE_THRESHOLD * drift * lookup(reference.scale[c])
        )

    return df.with_columns(
        is_spike(c).fill_null(False).alias(f"{c}_spike") for c in SENSOR_DATA
    )


def detect_corrupted(
    df: pl.DataFrame, reference: Reference | None = None
) -> pl.DataFrame:
    """Flag frozen and spiking sensor values.

    Call this on one chunk (for example a day of readings) at a time: the
    drift factor is estimated per machine over all of ``df``.

    Args:
        df: Data containing ``machine_id``, ``timestamp``, ``operating_mode``
            and every column in ``SENSOR_DATA``.
        reference: The reference to compare against; defaults to
            ``load_reference()``.

    Returns:
        ``df`` with ``<sensor>_frozen`` and ``<sensor>_spike`` columns added.
    """
    if reference is None:
        reference = load_reference()
    return flag_spikes(flag_frozen(df, reference), reference)


def remove_corrupted(
    df: pl.DataFrame, reference: Reference | None = None
) -> pl.DataFrame:
    """Flag corrupted readings, keep them aside, and null them for imputation.

    Args:
        df: One chunk, as for ``detect_corrupted``.
        reference: As for ``detect_corrupted``.

    Returns:
        ``df`` with the ``detect_corrupted`` flag columns and a
        ``<sensor>_original`` column per sensor, which holds the removed
        reading where that sensor was frozen or spiking and null elsewhere.
        Those readings are null in the sensor columns.
    """
    flagged = detect_corrupted(df, reference)

    def corrupted(c: str) -> pl.Expr:
        return pl.col(f"{c}_frozen") | pl.col(f"{c}_spike")

    return flagged.with_columns(
        pl.when(corrupted(c)).then(pl.col(c)).alias(f"{c}_original")
        for c in SENSOR_DATA
    ).with_columns(pl.when(~corrupted(c)).then(pl.col(c)).alias(c) for c in SENSOR_DATA)


def detect_remove_and_impute(
    df: pl.DataFrame, reference: Reference | None = None
) -> pl.DataFrame:
    """Production pipeline: remove corrupted readings, then impute all gaps.

    Corrupted and missing readings are both filled by ``detect_and_impute``,
    and its ``quality_flags`` column reports each one with its issue
    (``sensor_spike``, ``sensor_frozen`` or ``missing_value``) and original
    reading.

    Args:
        df: One chunk, as for ``detect_corrupted``.
        reference: As for ``detect_corrupted``.

    Returns:
        The output of ``detect_and_impute`` on ``remove_corrupted(df)``.
    """
    return detect_and_impute(remove_corrupted(df, reference))


if __name__ == "__main__":
    data = load_data(DATA_PATH)
    reference = load_reference()
    out = pl.concat(
        detect_corrupted(day, reference)
        for _, day in data.group_by(pl.col(TIMESTAMP).dt.date())
    )
    print(out.select(pl.col("^.*_(frozen|spike)$").sum()))
