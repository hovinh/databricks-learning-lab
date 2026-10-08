from datetime import date, datetime

import pandas as pd


def date_to_batch_id(d: date) -> int:
    return int(d.strftime("%Y%m%d"))


def batch_id_to_date(batch_id: int) -> date:
    return datetime.strptime(str(batch_id), "%Y%m%d").date()


def utc_now() -> pd.Timestamp:
    """Naive UTC timestamp for created_at columns (session time zone is UTC)."""
    return pd.Timestamp.now(tz="UTC").tz_localize(None)
