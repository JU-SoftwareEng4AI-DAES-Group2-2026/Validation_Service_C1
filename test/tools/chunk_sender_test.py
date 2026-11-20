"""Unit tests for tools/chunk_sender.py, using a small synthetic dataset instead of the real file."""

import importlib.util
import json
import sys
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import polars as pl
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "chunk_sender.py"
spec = importlib.util.spec_from_file_location("chunk_sender", SCRIPT)
chunk_sender = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chunk_sender)

START = datetime(2012, 1, 14)
YEAR3_START = datetime(2014, 1, 14)
YEAR3_END = datetime(2015, 1, 14)
MACHINES = ["M001", "M002"]


def make_day(day: datetime, temperature: float | None = 50.0) -> list[dict]:
    """24 hourly rows per machine for one day, plus label columns the sender must drop."""
    return [
        {
            "timestamp": day + timedelta(hours=h),
            "machine_id": m,
            "operating_mode": "NORMAL",
            "temperature": temperature,
            "vibration": 2.0,
            "pressure": 98.0,
            "rpm": 1500.0,
            "power": 210.0,
            "anomaly": False,
            "concept_drift_active": True,
        }
        for m in MACHINES
        for h in range(24)
    ]


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """Write a parquet with days before, inside and after year 3 and point the script at it."""
    rows = (
        make_day(START)  # year 1, must be excluded
        + make_day(YEAR3_START - timedelta(days=1))  # last day of year 2, excluded
        + make_day(YEAR3_START)  # first day of year 3
        + make_day(YEAR3_START + timedelta(days=1), temperature=None)  # year 3, missing temps
        + make_day(YEAR3_END - timedelta(days=1))  # last day of year 3
        + make_day(YEAR3_END)  # first day of year 4, excluded
    )
    df = pl.DataFrame(rows).with_columns(pl.col("machine_id", "operating_mode").cast(pl.Categorical))
    path = tmp_path / "data.parquet"
    df.write_parquet(path)
    monkeypatch.setattr(chunk_sender, "DATA_PATH", path)
    return path


def run_main(monkeypatch, *args: str) -> list[float]:
    """Run main() with CLI args; return the sleep durations instead of actually sleeping."""
    sleeps: list[float] = []
    monkeypatch.setattr(sys, "argv", ["chunk_sender.py", *args])
    monkeypatch.setattr(chunk_sender.time, "sleep", sleeps.append)
    chunk_sender.main()
    return sleeps


# --- load_year3 -------------------------------------------------------------


def test_load_year3_keeps_only_year_3(dataset):
    days = chunk_sender.load_year3()["day"].unique().sort().to_list()
    assert [str(d) for d in days] == ["2014-01-14", "2014-01-15", "2015-01-13"]


def test_load_year3_keeps_only_sensor_columns(dataset):
    df = chunk_sender.load_year3()
    assert df.columns == [*chunk_sender.COLUMNS, "day"]


def test_load_year3_formats_values_for_json(dataset):
    first = chunk_sender.load_year3().row(0, named=True)
    assert first["timestamp"] == "2014-01-14T00:00:00"
    assert first["machine_id"] == "M001"
    assert isinstance(first["operating_mode"], str)


# --- main, dry run ----------------------------------------------------------


def test_dry_run_sends_one_chunk_per_machine_per_day(dataset, monkeypatch, capsys):
    run_main(monkeypatch, "--dry-run")
    chunks = [json.loads(line) for line in capsys.readouterr().out.splitlines()]

    assert len(chunks) == 3 * len(MACHINES)
    for chunk in chunks:
        records = chunk["records"]
        assert len(records) == 24
        assert {r["machine_id"] for r in records} == {records[0]["machine_id"]}
        assert {r["timestamp"][:10] for r in records} == {records[0]["timestamp"][:10]}
        assert list(records[0]) == chunk_sender.COLUMNS
        assert [r["timestamp"] for r in records] == sorted(r["timestamp"] for r in records)


def test_missing_temperature_is_sent_as_null(dataset, monkeypatch, capsys):
    run_main(monkeypatch, "--dry-run")
    out = capsys.readouterr().out.splitlines()
    day2 = [json.loads(line) for line in out if '"2014-01-15T' in line]
    assert day2 and all(r["temperature"] is None for c in day2 for r in c["records"])


def test_days_limits_output_and_sleeps_between_days(dataset, monkeypatch, capsys):
    sleeps = run_main(monkeypatch, "--dry-run", "--days", "2", "--interval", "1.5")
    assert len(capsys.readouterr().out.splitlines()) == 2 * len(MACHINES)
    assert sleeps == [1.5]  # one pause between two days, none after the last


# --- post_chunk ---------------------------------------------------------------


@pytest.fixture
def server():
    """A local HTTP server that records POST bodies and answers with a configurable status."""
    received: list[dict] = []
    state = {"status": 202}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            received.append({"path": self.path, "body": json.loads(self.rfile.read(length))})
            self.send_response(state["status"])
            self.end_headers()

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1/chunks", received, state
    httpd.shutdown()


def test_post_chunk_sends_records_and_returns_status(server):
    url, received, _ = server
    records = [{"machine_id": "M001", "temperature": None}]

    assert chunk_sender.post_chunk(url, records) == "202"
    assert received == [{"path": "/v1/chunks", "body": {"records": records}}]


def test_post_chunk_reports_http_errors(server):
    url, _, state = server
    state["status"] = 422
    assert chunk_sender.post_chunk(url, [{}]) == "HTTP 422"


def test_post_chunk_reports_unreachable_server_without_raising():
    status = chunk_sender.post_chunk("http://127.0.0.1:9/v1/chunks", [{}])
    assert status.startswith("failed (")


def test_main_posts_every_chunk(dataset, server, monkeypatch, capsys):
    url, received, _ = server
    run_main(monkeypatch, "--url", url, "--interval", "0")

    assert len(received) == 3 * len(MACHINES)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "2014-01-14 M001: 24 records -> 202"
