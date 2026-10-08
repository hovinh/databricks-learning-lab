import sys
from pathlib import Path

# Bootstrap: put ml/ on sys.path. A Databricks spark_python_task execs this file
# without binding __file__, but its wrapper's `filename` local is visible here.
try:
    _entry = Path(__file__)
except NameError:
    _entry = Path(filename)  # noqa: F821
sys.path.insert(0, str(_entry.resolve().parent.parent))

import logging  # noqa: E402
from datetime import date, timedelta  # noqa: E402
from typing import Optional  # noqa: E402

import connect  # noqa: E402
import runtime  # noqa: E402
import settings  # noqa: E402
from features.batch import batch_id_to_date, date_to_batch_id, utc_now  # noqa: E402
from features.build import build_features  # noqa: E402
from utils.log import setup_logger  # noqa: E402

logger = logging.getLogger(__name__)


def next_unprocessed_date(env) -> date:
    """The simulated clock: the day after the latest feature batch, starting
    once there are FEATURE_LOOKBACK_DAYS of history behind the first day."""
    first_day, last_day = connect.read_source_date_range(env)
    latest = connect.read_latest_batch_id(env, settings.FEATURES_TABLE)
    if latest is None:
        candidate = first_day + timedelta(days=settings.FEATURE_LOOKBACK_DAYS)
    else:
        candidate = batch_id_to_date(latest) + timedelta(days=1)
    if candidate > last_day:
        raise RuntimeError(
            f"Simulated clock reached the end of the source data ({last_day}). "
            "Pass --batch-date to rebuild a specific day, or drop the features "
            "table to restart the simulation."
        )
    return candidate


def feature_engineering(env, batch_date: Optional[date] = None) -> int:
    """Builds and writes one feature batch. Returns its batch_id."""
    if batch_date is None:
        batch_date = next_unprocessed_date(env)
    logger.info("Building features for %s", batch_date)

    start = batch_date - timedelta(days=settings.FEATURE_LOOKBACK_DAYS)
    trips = connect.read_trips(env, start, batch_date)
    features = build_features(trips, batch_date, settings.FEATURE_LOOKBACK_DAYS)
    if features.empty:
        raise ValueError(f"No trips picked up on {batch_date}: nothing to write.")

    features["created_at"] = utc_now()
    connect.write_batch(env, features, settings.FEATURES_TABLE)

    batch_id = date_to_batch_id(batch_date)
    runtime.set_task_value("batch_id", batch_id)
    logger.info("Wrote %d feature rows for batch_id=%d", len(features), batch_id)
    return batch_id


def backfill(env, start: date, end: date) -> None:
    day = start
    while day <= end:
        feature_engineering(env, day)
        day += timedelta(days=1)


if __name__ == "__main__":
    parser = settings.base_arg_parser()
    parser.add_argument("--batch-date", type=settings.optional_date)
    parser.add_argument("--start-date", type=settings.optional_date)
    parser.add_argument("--end-date", type=settings.optional_date)
    args, _ = parser.parse_known_args()

    setup_logger(process_name="feature_engineering")
    env = settings.env_from_args(args)
    if args.start_date:
        backfill(env, args.start_date, args.end_date or args.start_date)
    else:
        feature_engineering(env, args.batch_date)
