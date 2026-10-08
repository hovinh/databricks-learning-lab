from datetime import date
from unittest.mock import patch

import pytest

from pipelines.feature_engineering import next_unprocessed_date

SOURCE_RANGE = (date(2016, 1, 1), date(2016, 2, 29))


def _next(env, latest_batch_id):
    with patch("connect.read_source_date_range", return_value=SOURCE_RANGE), patch(
        "connect.read_latest_batch_id", return_value=latest_batch_id
    ):
        return next_unprocessed_date(env)


def test_first_run_starts_after_the_lookback(uc_env):
    assert _next(uc_env, None) == date(2016, 1, 8)


def test_next_run_is_the_day_after_the_latest_batch(uc_env):
    assert _next(uc_env, 20160110) == date(2016, 1, 11)


def test_end_of_source_data_fails_loudly(uc_env):
    with pytest.raises(RuntimeError, match="end of the source data"):
        _next(uc_env, 20160229)
