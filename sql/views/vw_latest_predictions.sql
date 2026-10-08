CREATE OR REPLACE VIEW {catalog}.{schema}.vw_latest_predictions
COMMENT 'Most recent prediction batch next to the actual fare'
AS
SELECT
    p.trip_id
    , p.batch_id
    , p.pickup_ts
    , p.predicted_fare
    , f.fare_amount AS actual_fare
    , p.predicted_fare - f.fare_amount AS error
    , p.model_version
    , p.created_at
FROM {catalog}.{schema}.taxi_predictions AS p
JOIN {catalog}.{schema}.taxi_features AS f
    ON p.trip_id = f.trip_id AND p.batch_id = f.batch_id
WHERE p.batch_id = (SELECT max(batch_id) FROM {catalog}.{schema}.taxi_predictions);
