-- Column order must match settings.PREDICTION_TABLE_COLUMNS (tests/test_ddl_contract.py).
CREATE TABLE IF NOT EXISTS {catalog}.{schema}.taxi_predictions (
    trip_id           BIGINT     COMMENT 'Trip identifier, joins to taxi_features.trip_id'
    , batch_id        INT        COMMENT 'Batch the prediction was made for, as yyyyMMdd'
    , pickup_ts       TIMESTAMP  COMMENT 'Pickup timestamp (UTC)'
    , predicted_fare  DOUBLE     COMMENT 'Predicted fare in USD'
    , model_version   STRING     COMMENT 'Registered model version used, or local'
    , created_at      TIMESTAMP  COMMENT 'When the row was written (UTC)'
)
COMMENT 'Fare predictions, one row per trip per batch. Rewritten idempotently per batch_id by the predict pipeline.';
