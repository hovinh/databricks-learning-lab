from datetime import date

import pandas as pd

import settings
from features.build import build_features

BATCH_DAY = date(2016, 1, 10)


def _trip(pickup, fare, pickup_zip=10001, dropoff_zip=10002, distance=1.0):
    ts = pd.Timestamp(pickup)
    return {
        "tpep_pickup_datetime": ts,
        "tpep_dropoff_datetime": ts + pd.Timedelta(minutes=10),
        "trip_distance": distance,
        "fare_amount": fare,
        "pickup_zip": pickup_zip,
        "dropoff_zip": dropoff_zip,
    }


def test_only_batch_day_trips_are_returned():
    trips = pd.DataFrame(
        [_trip("2016-01-09 08:00", 10.0), _trip("2016-01-10 08:00", 12.0)]
    )
    out = build_features(trips, BATCH_DAY, lookback_days=7)
    assert len(out) == 1
    assert out["batch_id"].iloc[0] == 20160110


def test_pair_stats_use_only_days_before_the_batch():
    trips = pd.DataFrame(
        [
            _trip("2016-01-08 08:00", 10.0),
            _trip("2016-01-09 08:00", 20.0),
            _trip("2016-01-10 08:00", 999.0),  # the batch day's own fare: never used
        ]
    )
    out = build_features(trips, BATCH_DAY, lookback_days=7)
    assert out["pair_median_fare"].iloc[0] == 15.0
    assert out["pair_trip_count"].iloc[0] == 2


def test_history_outside_the_lookback_is_ignored():
    trips = pd.DataFrame(
        [_trip("2016-01-01 08:00", 10.0), _trip("2016-01-10 08:00", 12.0)]
    )
    out = build_features(trips, BATCH_DAY, lookback_days=7)
    assert pd.isna(out["pair_median_fare"].iloc[0])
    assert out["pair_trip_count"].iloc[0] == 0


def test_output_columns_match_the_feature_table():
    trips = pd.DataFrame([_trip("2016-01-10 08:00", 12.0)])
    out = build_features(trips, BATCH_DAY, lookback_days=7)
    expected = [c for c in settings.FEATURE_TABLE_COLUMNS if c != "created_at"]
    assert list(out.columns) == expected


def test_trip_id_is_deterministic():
    trips = pd.DataFrame([_trip("2016-01-10 08:00", 12.0)])
    first = build_features(trips, BATCH_DAY, 7)["trip_id"].iloc[0]
    second = build_features(trips, BATCH_DAY, 7)["trip_id"].iloc[0]
    assert first == second
