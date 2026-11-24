"""Detect and impute missing sensor readings in predictive-maintenance data.

The pipeline counts nulls in the sensor columns, flags each missing value in a
``<sensor>_imputed`` column, and fills the gaps with a per-machine median
computed from the same chunk of data. A ``quality_flags`` column then lists,
per row, each sensor value that was missing and what was done about it.
"""

from pathlib import Path

import polars as pl

# A list of the sensor data we are interested in
SENSOR_DATA = ["temperature", "vibration", "pressure", "rpm", "power"]

# Columns that identify which readings a missing value is filled from.
MACHINE = "machine_id"
MODE = "operating_mode"

# The dataset in the repository root.
DATA_PATH = Path(__file__).resolve().parents[2] / "predictive_maintenance_data.parquet"


def detect(df: pl.DataFrame) -> pl.DataFrame:
    """Count the missing values in each sensor column.

    Args:
        df: Data containing every column in ``SENSOR_DATA``.

    Returns:
        A one-row DataFrame with the null count of each sensor column.
    """
    null_count_df = df.select(SENSOR_DATA).null_count()
    # print(null_count_df)
    return null_count_df


def flag_missing(df: pl.DataFrame) -> pl.DataFrame:
    """Flag missing sensor values before they are imputed.

    Call this before ``impute`` so the flags record which values were
    originally missing.

    Args:
        df: Data containing every column in ``SENSOR_DATA``.

    Returns:
        ``df`` with a boolean ``<sensor>_imputed`` column added for each
        sensor, True where that sensor's value is null.
    """
    return df.with_columns(
        pl.col(c).is_null().alias(f"{c}_imputed") for c in SENSOR_DATA
    )


def impute(df: pl.DataFrame) -> pl.DataFrame:
    """Fill missing sensor values with medians from the same chunk.

    Each missing value is replaced by the median of that sensor for the same
    machine and operating mode. If the machine has no other readings in that
    mode, the machine's median over all modes is used instead. Only the rows
    in ``df`` are used, so no model or historical data is needed. A value
    stays null only if the machine has no readings at all for that sensor.

    Args:
        df: A chunk of data containing every column in ``SENSOR_DATA`` plus
            ``machine_id`` and ``operating_mode``.

    Returns:
        ``df`` with the nulls in the sensor columns filled in; other columns
        are untouched.
    """
    sensors = pl.col(SENSOR_DATA)
    return df.with_columns(
        sensors.fill_null(sensors.median().over(MACHINE, MODE)).fill_null(
            sensors.median().over(MACHINE)
        )
    )


def quality_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Describe each row's imputed sensor values in a ``quality_flags`` column.

    Call this after ``flag_missing`` and ``impute``. The column is a list of
    structs that serialises directly to the output JSON, e.g.::

        [{"sensor": "temperature", "issue": "missing_value",
          "action": "imputed", "original_value": null}]

    Rows with no imputed values get an empty list.

    If ``corrupted_measurements.remove_corrupted`` ran first, the
    ``<sensor>_spike`` / ``<sensor>_frozen`` and ``<sensor>_original``
    columns are present. A value it removed is then reported as
    ``"sensor_spike"`` or ``"sensor_frozen"`` with the removed reading as
    ``original_value``. Without those columns every imputed value is a
    ``"missing_value"``.

    Args:
        df: Imputed data with the ``<sensor>_imputed`` columns from
            ``flag_missing``.

    Returns:
        ``df`` with a ``quality_flags`` column added. ``action`` is
        ``"imputed"`` if the value was filled in and ``"not_imputed"`` if it
        is still null.
    """

    def optional(name: str, default: pl.Expr) -> pl.Expr:
        return pl.col(name) if name in df.columns else default

    def issue(c: str) -> pl.Expr:
        no = pl.lit(False)
        return (
            pl.when(optional(f"{c}_spike", no))
            .then(pl.lit("sensor_spike"))
            .when(optional(f"{c}_frozen", no))
            .then(pl.lit("sensor_frozen"))
            .otherwise(pl.lit("missing_value"))
        )

    flags = [
        pl.when(pl.col(f"{c}_imputed")).then(
            pl.struct(
                sensor=pl.lit(c),
                issue=issue(c),
                action=pl.when(pl.col(c).is_not_null())
                .then(pl.lit("imputed"))
                .otherwise(pl.lit("not_imputed")),
                original_value=optional(
                    f"{c}_original", pl.lit(None, dtype=pl.Float64)
                ),
            )
        )
        for c in SENSOR_DATA
    ]
    return df.with_columns(quality_flags=pl.concat_list(flags).list.drop_nulls())


def detect_and_impute(df: pl.DataFrame) -> pl.DataFrame:
    """Run the full detect-and-impute pipeline.

    Args:
        df: Data containing every column in ``SENSOR_DATA``.

    Returns:
        ``df`` with ``<sensor>_imputed`` flag columns and a ``quality_flags``
        column added, and the missing sensor values filled in.
    """
    detect(df)
    imputed_df = quality_flags(impute(flag_missing(df)))
    detect(imputed_df)
    return imputed_df


def load_data(path: str | Path = DATA_PATH) -> pl.DataFrame:
    """Load the predictive-maintenance dataset from a parquet file.

    Args:
        path: Path to the parquet file; defaults to the dataset in the repo
            root.

    Returns:
        The data, with ``machine_id`` and ``operating_mode`` as strings (the
        same type the API receives) instead of categoricals.
    """
    return pl.read_parquet(path).with_columns(pl.col(MACHINE, MODE).cast(pl.String))


# For testing purposes, this file can be executed on its own
if __name__ == "__main__":
    df = load_data()
    print(detect(df))
    print(detect(detect_and_impute(df)))
