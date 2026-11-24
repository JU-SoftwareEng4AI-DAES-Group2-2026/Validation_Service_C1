## receive_chunks.py

## Running the service

Replays year 3 of `predictive_maintenance_data.parquet` to `POST http://localhost:3333/v1/chunks`,
one chunk per machine per day (20 machines per day), with one day every 5 seconds.
The same options always send the same chunks in the same order.

Start the validation service first (it listens on port 3333):

```bash
uvicorn validation_service_c1.receive_chunks:app --app-dir src --port 3333
```

Then run the sender:

```bash
python tools/chunk_sender.py                     # all of year 3
...
```

The sender only prints the status code, so it doesn't show whether imputation worked. To confirm:

Open http://localhost:3333/docs, click POST /v1/chunks, "Try it out", and send the two-record example from before, with "temperature": null on the second record.
The response should show "status": "success", "imputed_per_sensor": {"temperature": 1, ...}, and "temperature_imputed": true in the second record.
Change "operating_mode" to "banana" and confirm you get the 422 with "status": "rejected" and the record_index/field details.