"""Everything that depends on *where* the code runs."""

import logging
import os

logger = logging.getLogger(__name__)


def is_databricks() -> bool:
    """True inside any Databricks runtime (job task, notebook), False locally."""
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def set_task_value(key: str, value) -> None:
    """Publishes a value to downstream tasks of the same job run, read with
    {{tasks.<task_key>.values.<key>}}. No-op outside Databricks."""
    if not is_databricks():
        return
    try:
        from databricks.sdk.runtime import dbutils

        dbutils.jobs.taskValues.set(key=key, value=value)
    except Exception as exc:  # e.g. run interactively, not as a job task
        logger.warning("Could not set task value %s=%s: %s", key, value, exc)


def get_widget(name: str, default: str = "") -> str:
    """For notebook tasks: a widget / base_parameter value, or `default` when
    it isn't set (interactive run, or not on Databricks at all)."""
    try:
        from databricks.sdk.runtime import dbutils

        return dbutils.widgets.get(name)
    except Exception:
        return default
