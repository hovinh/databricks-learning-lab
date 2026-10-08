from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

import settings
from pipelines.predict import predict


def _features(n=3):
    df = pd.DataFrame({c: np.arange(n, dtype="float64") for c in settings.FEATURES})
    df["trip_id"] = range(n)
    df["batch_id"] = 20160110
    df["pickup_ts"] = pd.Timestamp("2016-01-10 08:00")
    df[settings.TARGET] = 10.0
    return df


def _run(env, features, model):
    with patch("connect.read_features", return_value=(features, {})), patch(
        "mlflow_utils.load_champion", return_value=(model, "7")
    ), patch("connect.write_batch") as write:
        predict(env, batch_id=20160110)
    return write


def test_writes_prediction_table_columns(uc_env):
    model = MagicMock()
    model.predict.return_value = np.array([1.0, 2.0, 3.0])
    write = _run(uc_env, _features(), model)
    out = write.call_args[0][1]
    assert list(out.columns) == settings.PREDICTION_TABLE_COLUMNS
    assert out["predicted_fare"].tolist() == [1.0, 2.0, 3.0]
    assert out["model_version"].unique().tolist() == ["7"]


def test_model_receives_only_feature_columns(uc_env):
    model = MagicMock()
    model.predict.return_value = np.zeros(3)
    _run(uc_env, _features(), model)
    assert list(model.predict.call_args[0][0].columns) == settings.FEATURES


def test_missing_feature_fails_loudly(uc_env):
    with pytest.raises(ValueError, match="missing columns"):
        _run(uc_env, _features().drop(columns=["trip_distance"]), MagicMock())


def test_empty_batch_fails_loudly(uc_env):
    with pytest.raises(ValueError, match="No feature rows"):
        _run(uc_env, _features().iloc[0:0], MagicMock())
