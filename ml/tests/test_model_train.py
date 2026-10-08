import pandas as pd
import pytest

import settings
from model.train import clean_labels, prepare_X, time_split


def _batches(batch_ids, rows_per_batch=2):
    return pd.DataFrame(
        {"batch_id": [b for b in batch_ids for _ in range(rows_per_batch)]}
    )


def test_last_batches_are_the_test_set():
    df = _batches([20160101, 20160102, 20160103, 20160104])
    train_df, test_df = time_split(df, test_batches=2)
    assert sorted(train_df["batch_id"].unique()) == [20160101, 20160102]
    assert sorted(test_df["batch_id"].unique()) == [20160103, 20160104]


def test_too_few_batches_raises():
    with pytest.raises(ValueError, match="Backfill more"):
        time_split(_batches([20160101, 20160102]), test_batches=2)


def test_prepare_X_selects_features_as_float64():
    df = pd.DataFrame({c: [1] for c in settings.FEATURES})
    df["trip_id"] = 7
    X = prepare_X(df)
    assert list(X.columns) == settings.FEATURES
    assert (X.dtypes == "float64").all()


def test_clean_labels_drops_non_positive_fares():
    df = pd.DataFrame({settings.TARGET: [-1.0, 0.0, 5.0]})
    assert clean_labels(df)[settings.TARGET].tolist() == [5.0]
