"""Replay year 3 of the dataset to the validation API as per-machine chunks.

Each POST body is {"records": [...]} with one machine's hourly readings for one time
window (24 hours by default). All machines for a window are sent, then the script waits
--interval seconds before the next window.

    python tools/chunk_sender.py                       # send to localhost:3333, one day every 5 s
    python tools/chunk_sender.py --chunk-hours 6 --interval 1
    python tools/chunk_sender.py --machines M001 M002 --start 2014-03-01 --end 2014-03-31
    python tools/chunk_sender.py --dry-run --days 1    # print the chunks, send nothing
"""

import argparse
import json
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

DATA_PATH = Path(__file__).resolve().parent.parent / "predictive_maintenance_data.parquet"
DEFAULT_URL = "http://localhost:3333/v1/chunks"

COLUMNS = [
    "timestamp",
    "machine_id",
    "operating_mode",
    "temperature",
    "vibration",
    "pressure",
    "rpm",
    "power",
]


def load_year3() -> pl.DataFrame:
    """Load the sensor columns for year 3 of the dataset (2 to 3 years after the first timestamp)."""
    ts = pl.col("timestamp")
    return (
        pl.read_parquet(DATA_PATH, columns=COLUMNS)
        .filter((ts >= ts.min().dt.offset_by("2y")) & (ts < ts.min().dt.offset_by("3y")))
        .with_columns(pl.col("machine_id", "operating_mode").cast(pl.String))
        .sort("timestamp", "machine_id")
    )


def select(
    df: pl.DataFrame,
    machines: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    days: int | None = None,
) -> pl.DataFrame:
    """Keep only the given machines and the inclusive date range [start, end].

    `days` limits the range to that many days from `start` (or from the first day in the data).
    """
    if machines:
        unknown = set(machines) - set(df["machine_id"].unique())
        if unknown:
            raise ValueError(f"unknown machine(s): {', '.join(sorted(unknown))}")
        df = df.filter(pl.col("machine_id").is_in(machines))

    first_day = start or df["timestamp"].min().date()
    lower = datetime.combine(first_day, datetime.min.time())
    upper = datetime.combine(end, datetime.min.time()) + timedelta(days=1) if end else None
    if days is not None:
        days_upper = lower + timedelta(days=days)
        upper = min(upper, days_upper) if upper else days_upper

    df = df.filter(pl.col("timestamp") >= lower)
    if upper:
        df = df.filter(pl.col("timestamp") < upper)
    return df


def make_chunks(df: pl.DataFrame, chunk_hours: int = 24) -> list[list[list[dict]]]:
    """Split into time windows of `chunk_hours`; each window holds one record list per machine.

    Windows are aligned to the first timestamp in `df`, and everything is sorted, so the same
    input and settings always give the same chunks in the same order.
    """
    if chunk_hours < 1:
        raise ValueError("chunk_hours must be at least 1")
    if df.is_empty():
        return []

    start = df["timestamp"].min()
    df = df.with_columns(
        window=((pl.col("timestamp") - start).dt.total_hours() // chunk_hours),
        timestamp=pl.col("timestamp").dt.strftime("%Y-%m-%dT%H:%M:%S"),
    ).sort("window", "machine_id", "timestamp")

    return [
        [m.drop("window").to_dicts() for m in w.partition_by("machine_id", maintain_order=True)]
        for w in df.partition_by("window", maintain_order=True)
    ]


def post_chunk(url: str, records: list[dict]) -> str:
    """POST one chunk and return a short status string; never raises on HTTP/network errors."""
    body = json.dumps({"records": records}).encode()
    request = urllib.request.Request(
        url, data=body, method="POST", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return str(response.status)
    except urllib.error.HTTPError as exc:
        return f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return f"failed ({exc.reason})"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=DEFAULT_URL, help=f"chunk endpoint (default {DEFAULT_URL})")
    parser.add_argument("--chunk-hours", type=int, default=24, help="hours of readings per chunk (default 24)")
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between windows (default 5)")
    parser.add_argument("--machines", nargs="+", metavar="ID", help="only these machines, e.g. M001 M002")
    parser.add_argument("--start", type=date.fromisoformat, help="first day to send, YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, help="last day to send (inclusive), YYYY-MM-DD")
    parser.add_argument("--days", type=int, help="only send this many days")
    parser.add_argument("--dry-run", action="store_true", help="print chunks instead of sending them")
    args = parser.parse_args()

    try:
        df = select(load_year3(), args.machines, args.start, args.end, args.days)
        windows = make_chunks(df, args.chunk_hours)
    except ValueError as exc:
        parser.error(str(exc))
    if not windows:
        parser.error("no data matches the given machines/dates (year 3 is 2014-01-14 to 2015-01-12)")

    for i, window in enumerate(windows):
        for records in window:
            if args.dry_run:
                print(json.dumps({"records": records}))
            else:
                status = post_chunk(args.url, records)
                first = records[0]
                print(f"{first['timestamp']} {first['machine_id']}: {len(records)} records -> {status}")

        if i < len(windows) - 1:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
