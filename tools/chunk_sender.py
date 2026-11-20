"""Replay year 3 of the dataset to the validation API as per-machine, per-day chunks.

Each POST body is {"records": [...]} with one machine's 24 hourly readings for one day.
All machines for a day are sent together, then the script waits --interval seconds
before the next day.

    python tools/chunk_sender.py                     # send to localhost:3333, 5 s per day
    python tools/chunk_sender.py --interval 1 --days 3
    python tools/chunk_sender.py --dry-run --days 1  # print the chunks, send nothing
"""

import argparse
import json
import time
import urllib.error
import urllib.request
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
        .with_columns(
            pl.col("timestamp").dt.strftime("%Y-%m-%dT%H:%M:%S"),
            pl.col("machine_id", "operating_mode").cast(pl.String),
            day=pl.col("timestamp").dt.date(),
        )
        .sort("day", "machine_id", "timestamp")
    )


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
    parser.add_argument("--interval", type=float, default=5.0, help="seconds between days (default 5)")
    parser.add_argument("--days", type=int, help="only send this many days")
    parser.add_argument("--dry-run", action="store_true", help="print chunks instead of sending them")
    args = parser.parse_args()

    df = load_year3()
    days = df.partition_by("day", maintain_order=True, include_key=False)
    if args.days is not None:
        days = days[: args.days]

    for i, day_df in enumerate(days):
        day = day_df["timestamp"][0][:10]
        for machine_df in day_df.partition_by("machine_id", maintain_order=True):
            records = machine_df.to_dicts()
            machine_id = records[0]["machine_id"]
            if args.dry_run:
                print(json.dumps({"records": records}))
            else:
                status = post_chunk(args.url, records)
                print(f"{day} {machine_id}: {len(records)} records -> {status}")

        if i < len(days) - 1:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
