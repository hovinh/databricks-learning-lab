import sys
from pathlib import Path

import pytest

ML_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ML_ROOT))
sys.path.insert(0, str(ML_ROOT.parent / "sql" / "runner"))

import connect  # noqa: E402
import settings  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_local_dirs(tmp_path, monkeypatch):
    """Every local path -> tmp_path. Paths derived from another constant at
    import time don't follow a patched parent, so each is patched explicitly."""
    raw = tmp_path / "01_raw"
    monkeypatch.setattr(settings, "RAW_DIR", raw)
    monkeypatch.setattr(settings, "LOCAL_SOURCE_PATH", raw / "source_trips.parquet")
    monkeypatch.setattr(settings, "LOCAL_TABLES_DIR", tmp_path / "02_tables")
    monkeypatch.setattr(settings, "MODEL_DIR", tmp_path / "03_model")
    monkeypatch.setattr(settings, "LOCAL_MODEL_PATH", tmp_path / "03_model" / "m.txt")
    return tmp_path


@pytest.fixture(autouse=True)
def no_spark(monkeypatch):
    """Any test that reaches a real Spark session fails immediately."""

    def _fail():
        pytest.fail("Test tried to open a Spark session; mock the connect.* call")

    monkeypatch.setattr(connect, "get_spark", _fail)


@pytest.fixture(autouse=True)
def off_databricks(monkeypatch):
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)


@pytest.fixture
def local_env():
    return settings.Env(catalog="test_cat", schema="test_sch", data_source="local")


@pytest.fixture
def uc_env():
    return settings.Env(catalog="test_cat", schema="test_sch", data_source="uc")
