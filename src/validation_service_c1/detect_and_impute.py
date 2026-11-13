import io

import polars as pl
from sklearn.impute import KNNImputer

# A list, not a set: the order must stay fixed so the imputed numpy columns
# map back to the right names.
SENSOR_DATA = ["temperature", "vibration", "pressure", "rpm", "power"]


def detect(df: pl.DataFrame) -> pl.DataFrame:
    """Return the null count of each sensor column."""
    null_count_df = df.select(SENSOR_DATA).null_count()
    print(null_count_df)
    return null_count_df


def flag_missing(df: pl.DataFrame) -> pl.DataFrame:
    """Add a <sensor>_imputed column that is True where the value is missing."""
    return df.with_columns(
        pl.col(c).is_null().alias(f"{c}_imputed") for c in SENSOR_DATA
    )


def impute(df: pl.DataFrame) -> pl.DataFrame:
    """Fill nulls in the sensor columns with KNN; other columns are untouched."""
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    # KNNImputer only handles numeric data, so pass just the sensor columns.
    imputed = imputer.fit_transform(df.select(SENSOR_DATA).to_numpy())
    return df.with_columns(
        pl.Series(name, imputed[:, i]) for i, name in enumerate(SENSOR_DATA)
    )


def dai(df: pl.DataFrame) -> pl.DataFrame:
    """Detect missing sensor values, impute them, and verify none remain."""
    detect(df)
    imputed_df = impute(flag_missing(df))
    detect(imputed_df)
    return imputed_df


def load_data(path: str) -> pl.DataFrame:
    # The .xlsx holds CSV text in a single column, so rebuild and parse it.
    raw = pl.read_excel(path)
    header = raw.columns[0]
    text = header + "\n" + "\n".join(raw[header].to_list())
    return pl.read_csv(io.StringIO(text), try_parse_dates=True)


if __name__ == "__main__":
    df = load_data("predictive_maintenance_data.xlsx")
    dai(df)
