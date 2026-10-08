"""The only module that talks to Spark / Unity Catalog.

Every public function takes `env` (settings.Env) first. With
env.data_source == "local", reads and writes go to parquet files under ml/data/
instead, so the pipelines run with no Databricks connection at all.
"""

import decimal
import functools
import logging
from datetime import date
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd

import runtime
import settings

logger = logging.getLogger(__name__)

_spark = None


def get_spark():
    """Creates the Spark session on first use, never at import time, so tests
    can import this module offline."""
    global _spark
    if _spark is None:
        if runtime.is_databricks():
            from pyspark.sql import SparkSession

            _spark = SparkSession.builder.getOrCreate()
        else:
            from databricks.connect import DatabricksSession

            _spark = (
                DatabricksSession.builder.serverless()
                .profile(settings.DATABRICKS_PROFILE)
                .getOrCreate()
            )
        # Same time zone locally and in jobs, so dates don't shift.
        _spark.conf.set("spark.sql.session.timeZone", "UTC")
    return _spark


def to_pandas(spark_df, limit: Optional[int] = None) -> pd.DataFrame:
    """spark_df.toPandas() plus the clean-up every caller would otherwise need:
    UC DECIMAL columns arrive as object dtype holding decimal.Decimal (LightGBM
    rejects them) -> float64; Databricks Connect fills df.attrs with query-plan
    metrics that to_parquet() can't serialise -> cleared."""
    if limit is not None:
        spark_df = spark_df.limit(limit)
    df = spark_df.toPandas()
    df.attrs = {}
    for col in df.columns:
        if df[col].dtype != object:
            continue
        non_null = df[col].dropna()
        if not non_null.empty and isinstance(non_null.iloc[0], decimal.Decimal):
            df[col] = df[col].astype("float64")
    return df


# --- local debug snapshots ---------------------------------------------------


def _write_snapshot(df: pd.DataFrame, name: str) -> None:
    settings.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.RAW_DIR / f"{name}.csv", index=False)
    df.to_parquet(settings.RAW_DIR / f"{name}.parquet", index=False)
    logger.info("Snapshot of %s written to %s", name, settings.RAW_DIR)


def export_snapshot(name: Optional[str] = None, selector: Optional[Callable] = None):
    """Also dumps the decorated read's result to ml/data/01_raw/<name>.csv and
    .parquet when running locally against UC. The decorated function must take
    `env` as its first argument; `selector` picks the DataFrame out of a tuple
    result."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(env, *args, **kwargs):
            result = func(env, *args, **kwargs)
            if not runtime.is_databricks() and env.data_source == "uc":
                df = selector(result) if selector else result
                _write_snapshot(df, name or func.__name__)
            return result

        return wrapper

    return decorator


# --- local "tables" (data_source == "local") ---------------------------------


def _local_table_path(table_name: str) -> Path:
    return settings.LOCAL_TABLES_DIR / f"{table_name}.parquet"


def _read_local_table(table_name: str) -> pd.DataFrame:
    path = _local_table_path(table_name)
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _read_local_source() -> pd.DataFrame:
    if not settings.LOCAL_SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"No local source data at {settings.LOCAL_SOURCE_PATH}. Run "
            "`python -m scripts.export_source` once, or put your own extract there."
        )
    return pd.read_parquet(settings.LOCAL_SOURCE_PATH)


# --- reads ---------------------------------------------------------------------


@export_snapshot()
def read_trips(env, start: date, end: date) -> pd.DataFrame:
    """Source trips picked up between start and end (inclusive)."""
    if env.data_source == "local":
        df = _read_local_source()
        pickup_date = df["tpep_pickup_datetime"].dt.date
        return df[(pickup_date >= start) & (pickup_date <= end)].reset_index(drop=True)
    spark_df = (
        get_spark()
        .table(settings.SOURCE_TABLE)
        .filter(f"to_date(tpep_pickup_datetime) BETWEEN DATE'{start}' AND DATE'{end}'")
    )
    df = to_pandas(spark_df)
    logger.info("Read %d trips picked up %s..%s", len(df), start, end)
    return df


def read_source_date_range(env) -> Tuple[date, date]:
    """First and last pickup date in the source data."""
    if env.data_source == "local":
        pickup_date = _read_local_source()["tpep_pickup_datetime"].dt.date
        return pickup_date.min(), pickup_date.max()
    row = (
        get_spark()
        .table(settings.SOURCE_TABLE)
        .selectExpr(
            "min(to_date(tpep_pickup_datetime)) AS first_day",
            "max(to_date(tpep_pickup_datetime)) AS last_day",
        )
        .collect()[0]
    )
    return row["first_day"], row["last_day"]


def read_latest_batch_id(env, table_name: str) -> Optional[int]:
    """Max batch_id in the table, or None when it's empty."""
    if env.data_source == "local":
        df = _read_local_table(table_name)
        return None if df.empty else int(df["batch_id"].max())
    row = (
        get_spark()
        .table(env.table(table_name))
        .selectExpr("max(batch_id) AS b")
        .collect()[0]
    )
    return None if row["b"] is None else int(row["b"])


def read_table_version(env, table_name: str) -> Optional[int]:
    """Current Delta version. Logged with trained models so their exact input
    can be replayed with `SELECT * FROM t VERSION AS OF <n>`."""
    if env.data_source == "local":
        return None
    history = get_spark().sql(f"DESCRIBE HISTORY {env.table(table_name)} LIMIT 1")
    return int(history.collect()[0]["version"])


@export_snapshot(selector=lambda result: result[0])
def read_features(env, start_batch_id: int, end_batch_id: int):
    """Feature rows with batch_id in [start, end], plus a provenance dict."""
    provenance = {
        "table": env.table(settings.FEATURES_TABLE),
        "start_batch_id": start_batch_id,
        "end_batch_id": end_batch_id,
        "table_version": read_table_version(env, settings.FEATURES_TABLE),
    }
    if env.data_source == "local":
        df = _read_local_table(settings.FEATURES_TABLE)
        if not df.empty:
            df = df[df["batch_id"].between(start_batch_id, end_batch_id)]
        return df.reset_index(drop=True), provenance
    spark_df = (
        get_spark()
        .table(env.table(settings.FEATURES_TABLE))
        .filter(f"batch_id BETWEEN {start_batch_id} AND {end_batch_id}")
    )
    return to_pandas(spark_df), provenance


# --- writes --------------------------------------------------------------------


def write_batch(env, df: pd.DataFrame, table_name: str, batch_col: str = "batch_id"):
    """Writes exactly one batch, idempotently: that batch's rows are replaced and
    every other batch is untouched, so reruns never duplicate rows.

    UC mode aligns the DataFrame to the target table: columns are selected,
    ordered and cast from the table's own schema, and a missing column fails
    loudly. Callers never hand-maintain column order or int64-vs-INT casts."""
    batch_ids = df[batch_col].unique()
    if len(batch_ids) != 1:
        raise ValueError(f"write_batch expects one {batch_col}, got {list(batch_ids)}")
    batch_id = int(batch_ids[0])

    if env.data_source == "local":
        path = _local_table_path(table_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = _read_local_table(table_name)
        if not existing.empty:
            existing = existing[existing[batch_col] != batch_id]
            df = pd.concat([existing, df], ignore_index=True)
        df.to_parquet(path, index=False)
        logger.info("Wrote %s=%d to local table %s", batch_col, batch_id, path)
        return

    from pyspark.sql import functions as F

    spark = get_spark()
    table = env.table(table_name)
    fields = spark.table(table).schema.fields
    missing = [f.name for f in fields if f.name not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing columns of {table}: {missing}")
    extra = sorted(set(df.columns) - {f.name for f in fields})
    if extra:
        logger.warning("Ignoring columns not in %s: %s", table, extra)

    spark_df = spark.createDataFrame(df[[f.name for f in fields]])
    spark_df = spark_df.select(
        *[F.col(f.name).cast(f.dataType).alias(f.name) for f in fields]
    )
    (
        spark_df.write.mode("overwrite")
        .option("replaceWhere", f"{batch_col} = {batch_id}")
        .saveAsTable(table)
    )
    logger.info("Wrote %d rows to %s for %s=%d", len(df), table, batch_col, batch_id)
