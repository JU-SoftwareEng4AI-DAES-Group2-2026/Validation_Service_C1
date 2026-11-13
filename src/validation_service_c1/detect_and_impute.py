"""Detect and impute missing sensor readings in predictive-maintenance data.

The pipeline counts nulls in the sensor columns, flags each missing value in a
``<sensor>_imputed`` column, and fills the gaps with a per-machine median
computed from the same chunk of data.
"""

import io
from pathlib import Path

import polars as pl

# A list of the sensor data we are interested in
SENSOR_DATA = ["temperature", "vibration", "pressure", "rpm", "power"]

# Columns that identify which readings a missing value is filled from.
MACHINE = "machine_id"
MODE = "operating_mode"


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


def detect_and_impute(df: pl.DataFrame) -> pl.DataFrame:
    """Run the full detect-and-impute pipeline.

    Args:
        df: Data containing every column in ``SENSOR_DATA``.

    Returns:
        ``df`` with ``<sensor>_imputed`` flag columns added and the missing
        sensor values filled in.
    """
    detect(df)
    imputed_df = impute(flag_missing(df))
    detect(imputed_df)
    return imputed_df


def load_data(path: str | Path) -> pl.DataFrame:
    """Load the predictive-maintenance dataset from an .xlsx file.

    The workbook stores CSV text in a single column (the header line is the
    column name), so the rows are joined back into CSV text and parsed.

    Args:
        path: Path to the .xlsx file.

    Returns:
        The parsed data, with date columns converted to datetimes.
    """
    raw = pl.read_excel(path)
    header = raw.columns[0]
    text = header + "\n" + "\n".join(raw[header].to_list())
    return pl.read_csv(io.StringIO(text), try_parse_dates=True)


# For testing purposes, this file can be executed on its own
# with the provided xlsx sheet in the working directory
if __name__ == "__main__":
    df = load_data("predictive_maintenance_data.xlsx")
    detect_and_impute(df)
