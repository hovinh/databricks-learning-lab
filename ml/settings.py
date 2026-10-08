"""Single source of truth for names, features, local paths and the runtime
environment (which catalog/schema/data source a run targets)."""

import argparse
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

ML_ROOT = Path(__file__).resolve().parent

# Local-only paths (Stage 0 data, debug snapshots, local model). Gitignored.
DATA_DIR = ML_ROOT / "data"
RAW_DIR = DATA_DIR / "01_raw"
LOCAL_SOURCE_PATH = RAW_DIR / "source_trips.parquet"
LOCAL_TABLES_DIR = DATA_DIR / "02_tables"
MODEL_DIR = DATA_DIR / "03_model"
LOCAL_MODEL_PATH = MODEL_DIR / "model.txt"

# Same env var the Databricks SDK itself honours.
DATABRICKS_PROFILE = os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT")

SOURCE_TABLE = "samples.nyctaxi.trips"
FEATURES_TABLE = "taxi_features"
PREDICTIONS_TABLE = "taxi_predictions"
MODEL_BASENAME = "taxi_fare_model"
CHAMPION_ALIAS = "champion"

FEATURE_LOOKBACK_DAYS = 7  # history window for the zip-pair aggregates
TRAIN_WINDOW_DAYS = 28  # batches used per training run
TEST_BATCHES = 7  # most recent batches held out as the test set

TARGET = "fare_amount"
FEATURES = [
    "trip_distance",
    "pickup_zip",
    "dropoff_zip",
    "pickup_hour_sin",
    "pickup_hour_cos",
    "pickup_dow_sin",
    "pickup_dow_cos",
    "is_weekend",
    "pair_median_fare",
    "pair_trip_count",
]

# Must match sql/tables/*.sql column-for-column, in order
# (enforced by tests/test_ddl_contract.py).
FEATURE_TABLE_COLUMNS = [
    "trip_id",
    "batch_id",
    "pickup_ts",
    "pickup_zip",
    "dropoff_zip",
    "trip_distance",
    "pickup_hour_sin",
    "pickup_hour_cos",
    "pickup_dow_sin",
    "pickup_dow_cos",
    "is_weekend",
    "pair_median_fare",
    "pair_trip_count",
    "fare_amount",
    "created_at",
]
PREDICTION_TABLE_COLUMNS = [
    "trip_id",
    "batch_id",
    "pickup_ts",
    "predicted_fare",
    "model_version",
    "created_at",
]


@dataclass(frozen=True)
class Env:
    """Where a run reads and writes. Passed explicitly to every connect.* and
    mlflow_utils.* call instead of living in import-time globals."""

    catalog: str
    schema: str
    data_source: str = "uc"  # "uc" or "local"

    def table(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}"

    @property
    def model_name(self) -> str:
        return self.table(MODEL_BASENAME)

    @property
    def experiment_path(self) -> str:
        # One experiment per environment so dev and prd runs never mix.
        return f"/Shared/nyc_taxi_{self.schema}"


def optional_date(value: Optional[str]) -> Optional[date]:
    return date.fromisoformat(value) if value else None


def optional_int(value: Optional[str]) -> Optional[int]:
    # Empty string = an unresolved job/task-value reference; treat as "not given".
    return int(value) if value else None


def base_arg_parser() -> argparse.ArgumentParser:
    """Arguments every pipeline accepts. Resolution order: CLI argument (what
    jobs pass) -> env var (handy locally) -> default (dev)."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog", default=os.environ.get("TAXI_CATALOG", "workspace")
    )
    parser.add_argument("--schema", default=os.environ.get("TAXI_SCHEMA", "taxi_dev"))
    parser.add_argument(
        "--data-source",
        choices=["uc", "local"],
        default=os.environ.get("TAXI_DATA_SOURCE", "uc"),
    )
    return parser


def env_from_args(args: argparse.Namespace) -> Env:
    return Env(catalog=args.catalog, schema=args.schema, data_source=args.data_source)
