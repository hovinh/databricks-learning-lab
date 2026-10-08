from datetime import date, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import connect
import settings
from features.batch import date_to_batch_id
from pipelines.train import train


def _feature_batch(batch_date, n=40, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.random(n) for c in settings.FEATURES})
    df["trip_distance"] = rng.uniform(0.5, 10, n)
    df["trip_id"] = np.arange(n)
    df["batch_id"] = date_to_batch_id(batch_date)
    df["pickup_ts"] = pd.Timestamp(batch_date)
    df[settings.TARGET] = 2.5 + 2.0 * df["trip_distance"]
    df["created_at"] = pd.Timestamp("2026-01-01")
    return df


def _write_local_batches(env, n_batches):
    first = date(2016, 1, 8)
    for i in range(n_batches):
        batch = _feature_batch(first + timedelta(days=i), seed=i)
        connect.write_batch(env, batch, settings.FEATURES_TABLE)


def test_local_mode_saves_a_model_file(local_env):
    _write_local_batches(local_env, settings.TEST_BATCHES + 3)
    assert train(local_env) == "local"
    assert settings.LOCAL_MODEL_PATH.exists()


def test_empty_features_table_fails_loudly(uc_env):
    with patch("connect.read_latest_batch_id", return_value=None):
        with pytest.raises(ValueError, match="backfill first"):
            train(uc_env)


def test_uc_mode_registers_with_feature_only_test_frame(uc_env):
    batches = [
        _feature_batch(date(2016, 1, 8) + timedelta(days=i), seed=i)
        for i in range(settings.TEST_BATCHES + 3)
    ]
    features = pd.concat(batches, ignore_index=True)
    with patch("connect.read_features", return_value=(features, {"t": 1})), patch(
        "mlflow_utils.log_and_register", return_value="3"
    ) as register:
        assert train(uc_env, end_batch_id=20160117) == "3"
    X_test = register.call_args[0][2]
    assert list(X_test.columns) == settings.FEATURES
