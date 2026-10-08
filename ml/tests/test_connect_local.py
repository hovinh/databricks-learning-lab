import pandas as pd
import pytest

import connect


def _batch(batch_id, n):
    return pd.DataFrame({"batch_id": [batch_id] * n, "value": range(n)})


def test_write_batch_is_idempotent(local_env):
    connect.write_batch(local_env, _batch(20160110, 3), "t")
    connect.write_batch(local_env, _batch(20160110, 3), "t")
    assert len(connect._read_local_table("t")) == 3


def test_write_batch_keeps_other_batches(local_env):
    connect.write_batch(local_env, _batch(20160110, 3), "t")
    connect.write_batch(local_env, _batch(20160111, 2), "t")
    connect.write_batch(local_env, _batch(20160110, 1), "t")
    counts = connect._read_local_table("t")["batch_id"].value_counts().to_dict()
    assert counts == {20160110: 1, 20160111: 2}


def test_write_batch_rejects_mixed_batches(local_env):
    df = pd.DataFrame({"batch_id": [20160110, 20160111]})
    with pytest.raises(ValueError):
        connect.write_batch(local_env, df, "t")


def test_latest_batch_id_of_missing_table_is_none(local_env):
    assert connect.read_latest_batch_id(local_env, "does_not_exist") is None
