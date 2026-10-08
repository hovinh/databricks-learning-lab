"""Pure feature logic: pandas in, pandas out. No I/O."""

from datetime import date, timedelta

import numpy as np
import pandas as pd

import settings
from features.batch import date_to_batch_id

PAIR_KEYS = ["pickup_zip", "dropoff_zip"]
ID_SOURCE_COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "pickup_zip",
    "dropoff_zip",
    "trip_distance",
    "fare_amount",
]
OUTPUT_COLUMNS = [c for c in settings.FEATURE_TABLE_COLUMNS if c != "created_at"]


def _trip_id(df: pd.DataFrame) -> pd.Series:
    """Deterministic id from the trip's own values (the source has no key), so
    reruns produce the same ids. Identical duplicate trips share an id."""
    hashed = pd.util.hash_pandas_object(df[ID_SOURCE_COLUMNS], index=False)
    return (hashed % (2**63)).astype("int64")


def build_features(trips: pd.DataFrame, batch_date: date, lookback_days: int):
    """Features for the trips picked up on batch_date.

    Zip-pair aggregates use only trips from the `lookback_days` days *before*
    batch_date, never batch_date itself: a batch's features don't depend on
    when they were computed, and never contain the fares being predicted."""
    if trips.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    pickup_date = trips["tpep_pickup_datetime"].dt.date
    day = trips[pickup_date == batch_date].copy()
    history = trips[
        (pickup_date < batch_date)
        & (pickup_date >= batch_date - timedelta(days=lookback_days))
    ]

    pair_stats = (
        history.groupby(PAIR_KEYS)["fare_amount"]
        .agg(pair_median_fare="median", pair_trip_count="count")
        .reset_index()
    )
    day = day.merge(pair_stats, on=PAIR_KEYS, how="left")
    day["pair_trip_count"] = day["pair_trip_count"].fillna(0).astype("int64")

    hour = day["tpep_pickup_datetime"].dt.hour
    dow = day["tpep_pickup_datetime"].dt.dayofweek
    day["pickup_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    day["pickup_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    day["pickup_dow_sin"] = np.sin(2 * np.pi * dow / 7)
    day["pickup_dow_cos"] = np.cos(2 * np.pi * dow / 7)
    day["is_weekend"] = (dow >= 5).astype("int64")

    day["trip_id"] = _trip_id(day)
    day["batch_id"] = date_to_batch_id(batch_date)
    day = day.rename(columns={"tpep_pickup_datetime": "pickup_ts"})
    return day[OUTPUT_COLUMNS].reset_index(drop=True)
