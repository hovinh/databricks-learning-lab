"""Pure model logic: no I/O, no MLflow."""

import lightgbm as lgb
import numpy as np
import pandas as pd

import settings

PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    "verbose": -1,
}
NUM_BOOST_ROUND = 1000
EARLY_STOPPING_ROUNDS = 50


def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """Features only, all float64: the same frame for training, the MLflow
    signature, and prediction. Ints with/without nulls can't diverge."""
    return df[settings.FEATURES].astype("float64")


def clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows whose label can't be a real fare."""
    return df[df[settings.TARGET] > 0]


def time_split(df: pd.DataFrame, test_batches: int):
    """The most recent `test_batches` batches are the test set: no random
    split for time-ordered data."""
    batches = sorted(df["batch_id"].unique())
    if len(batches) <= test_batches:
        raise ValueError(
            f"Need more than {test_batches} batches to split, got {len(batches)}. "
            "Backfill more feature batches first."
        )
    is_test = df["batch_id"].isin(batches[-test_batches:])
    return df[~is_test], df[is_test]


def fit(train_df: pd.DataFrame, test_df: pd.DataFrame, params=None) -> lgb.Booster:
    params = params or PARAMS
    dtrain = lgb.Dataset(prepare_X(train_df), label=train_df[settings.TARGET])
    dvalid = lgb.Dataset(
        prepare_X(test_df), label=test_df[settings.TARGET], reference=dtrain
    )
    return lgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )


def evaluate(booster, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """MAE/RMSE on the test set, next to a naive baseline: the zip pair's recent
    median fare (falling back to the overall training median)."""
    y = test_df[settings.TARGET].to_numpy()
    pred = booster.predict(prepare_X(test_df))
    fallback = float(train_df[settings.TARGET].median())
    baseline = test_df["pair_median_fare"].fillna(fallback).to_numpy()
    return {
        "mae": float(np.mean(np.abs(pred - y))),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "baseline_mae": float(np.mean(np.abs(baseline - y))),
        "n_train": float(len(train_df)),
        "n_test": float(len(test_df)),
    }
