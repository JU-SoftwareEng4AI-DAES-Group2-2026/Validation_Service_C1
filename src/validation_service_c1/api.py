import logging
from contextlib import asynccontextmanager

import httpx
import polars as pl
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .detect_and_impute import SENSOR_DATA, detect_and_impute
from .schema import SensorBatch

logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(fastapi: FastAPI):
    async with httpx.AsyncClient(timeout=10.0) as client:
        fastapi.state.http_client = client
        yield


app = FastAPI(title="F1 - Chunk Validation Service", lifespan=lifespan)

# Explicit schema so a sensor that is entirely null in a chunk still becomes
# Float64 (otherwise Polars infers a Null dtype and the median step breaks).
CHUNK_SCHEMA = {
    "timestamp": pl.Datetime,
    "machine_id": pl.String,
    "operating_mode": pl.String,
    **{sensor: pl.Float64 for sensor in SENSOR_DATA},
}


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """FAILURE 1 (422): the chunk does not match the schema, so M2 is never called.

    Turns FastAPI's raw error list into one readable line per problem, e.g.
    "record 1, field 'operating_mode': Input should be 'HIGH_LOAD', 'NORMAL', ..."
    """
    problems = []
    for err in exc.errors():
        loc = err["loc"]
        record_index = (
            loc[2]
            if len(loc) >= 3
            and loc[:2] == ("body", "records")
            and isinstance(loc[2], int)
            else None
        )
        field = loc[-1] if loc and isinstance(loc[-1], str) else "body"
        where = (
            f"record {record_index}, field '{field}'"
            if record_index is not None
            else f"'{field}'"
        )
        problems.append(
            {
                "record_index": record_index,
                "field": str(field),
                "error_type": err["type"],
                "problem": f"{where}: {err['msg']}",
            }
        )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "status": "rejected",
            "stage": "validation",
            "message": f"Chunk rejected: {len(problems)} validation error(s). "
            "Nothing was passed to M2.",
            "errors": problems,
        },
    )


@app.post("/v1/chunks")
def receive_chunk(batch: SensorBatch) -> dict:
    """F1: accept a chunk, validate it, pass it to M2 to impute missing values.

    1. FastAPI validates the body against SensorBatch (failure -> 422 above).
    2. The valid chunk goes to M2 (failure -> 500 below).
    3. On success, returns a summary plus the imputed records.
    """
    try:
        rows = [record.model_dump() for record in batch.records]
        df = pl.DataFrame(rows, schema=CHUNK_SCHEMA)
        imputed = detect_and_impute(df)  # M2
    except Exception as exc:
        # FAILURE 2 (500): the data was valid, but processing it crashed.
        logger.exception("M2 imputation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "status": "failed",
                "stage": "imputation (M2)",
                "message": "Chunk was valid, but imputing missing values failed.",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )

    # SUCCESS: validated and imputed.
    still_null = imputed.select(SENSOR_DATA).null_count().row(0, named=True)
    return {
        "status": "success",
        "message": f"Chunk validated and imputed: {len(batch.records)} record(s).",
        "accepted": len(batch.records),
        "machines": sorted({r.machine_id for r in batch.records}),
        # Number of values M2 filled in per sensor (sum of the *_imputed flags).
        "imputed_per_sensor": {
            sensor: int(imputed[f"{sensor}_imputed"].sum()) - still_null[sensor]
            for sensor in SENSOR_DATA
        },
        # Gaps M2 could not fill (machine had no readings for that sensor in this chunk).
        "still_null": still_null,
        "records": imputed.to_dicts(),
    }


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("v1/model")
async def get_deployed_model() -> dict:
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented yet")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "validation_service_c1.receive_chunks:app",
        host="127.0.0.1",
        port=3333,
    )
