-- Column order must match settings.FEATURE_TABLE_COLUMNS (tests/test_ddl_contract.py).
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.taxi_features (
    trip_id             BIGINT     COMMENT 'Deterministic hash of the trip values (the source has no key)'
    , batch_id          INT        COMMENT 'Simulated batch date as yyyyMMdd, equal to the pickup date'
    , pickup_ts         TIMESTAMP  COMMENT 'Pickup timestamp (UTC)'
    , pickup_zip        INT        COMMENT 'Pickup zip code'
    , dropoff_zip       INT        COMMENT 'Dropoff zip code'
    , trip_distance     DOUBLE     COMMENT 'Trip distance in miles'
    , pickup_hour_sin   DOUBLE     COMMENT 'Pickup hour of day, sine encoded'
    , pickup_hour_cos   DOUBLE     COMMENT 'Pickup hour of day, cosine encoded'
    , pickup_dow_sin    DOUBLE     COMMENT 'Pickup day of week, sine encoded'
    , pickup_dow_cos    DOUBLE     COMMENT 'Pickup day of week, cosine encoded'
    , is_weekend        INT        COMMENT '1 if picked up on Saturday or Sunday'
    , pair_median_fare  DOUBLE     COMMENT 'Median fare for this zip pair over the 7 days before the batch date'
    , pair_trip_count   INT        COMMENT 'Number of trips for this zip pair over the 7 days before the batch date'
    , fare_amount       DOUBLE     COMMENT 'Label: actual fare in USD'
    , created_at        TIMESTAMP  COMMENT 'When the row was written (UTC)'
)
COMMENT 'One row per trip per batch. Rewritten idempotently per batch_id by the feature_engineering pipeline.';
