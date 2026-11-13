"""Detect and impute missing sensor readings in predictive-maintenance data.

The pipeline counts nulls in the sensor columns, flags each missing value in a
``<sensor>_imputed`` column, and fills the gaps with a KNN imputer.
"""

import io
from pathlib import Path

import polars as pl
from sklearn.impute import KNNImputer

# A list of the sensor data we are interested in
SENSOR_DATA = ["temperature", "vibration", "pressure", "rpm", "power"]


def detect(df: pl.DataFrame) -> pl.DataFrame:
    """Count the missing values in each sensor column.

    The counts are also printed so the pipeline can be followed from the
    console.

    Args:
        df: Data containing every column in ``SENSOR_DATA``.

    Returns:
        A one-row DataFrame with the null count of each sensor column.
    """
    null_count_df = df.select(SENSOR_DATA).null_count()
    print(null_count_df)
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
    """Fill missing sensor values with a KNN imputer.

    Each missing value is replaced by the mean of that sensor in the two
    nearest rows, measured over the sensor columns only. Non-sensor columns
    are left untouched.

    Args:
        df: Data containing every column in ``SENSOR_DATA``.

    Returns:
        ``df`` with the nulls in the sensor columns filled in. The sensor
        columns are returned as floats.
    """
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    # KNNImputer only handles numeric data, so pass just the sensor columns.
    imputed = imputer.fit_transform(df.select(SENSOR_DATA).to_numpy())
    return df.with_columns(
        pl.Series(name, imputed[:, i]) for i, name in enumerate(SENSOR_DATA)
    )


def dai(df: pl.DataFrame) -> pl.DataFrame:
    """Run the full detect-and-impute pipeline.

    Prints the null counts before and after imputation, so the second print
    should show zero for every sensor.

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
    dai(df)
