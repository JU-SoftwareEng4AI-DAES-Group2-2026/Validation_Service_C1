# tools/

## chunk_sender.py

Replays year 3 of `predictive_maintenance_data.parquet` to `POST http://localhost:3333/v1/chunks`,
one chunk per machine per day (20 machines per day), with one day every 5 seconds.

```bash
python tools/chunk_sender.py                     # all of year 3
python tools/chunk_sender.py --interval 1        # one day per second
python tools/chunk_sender.py --days 3            # only the first 3 days
python tools/chunk_sender.py --dry-run --days 1  # print the JSON, send nothing
python tools/chunk_sender.py --url http://localhost:8000/v1/chunks # Change the url from its preset
```

Each chunk has one machine's 24 hourly readings for one day:

```json
{
  "records": [
    {
      "timestamp": "2014-01-14T00:00:00",
      "machine_id": "M001",
      "operating_mode": "NORMAL",
      "temperature": 62.30627272064564,
      "vibration": 7.289666888263661,
      "pressure": 93.29908467406555,
      "rpm": 1621.8620237298148,
      "power": 236.9834106040295
    }
  ]
}
```

`operating_mode` is one of `NORMAL`, `HIGH_LOAD`, `IDLE` or `OVERLOAD`.
`temperature` can be `null`.
