import sys
from pathlib import Path

try:
    _entry = Path(__file__)
except NameError:
    _entry = Path(filename)  # noqa: F821
sys.path.insert(0, str(_entry.resolve().parent.parent))

import logging  # noqa: E402
from datetime import timedelta  # noqa: E402
from typing import Optional  # noqa: E402

import connect  # noqa: E402
import mlflow_utils  # noqa: E402
import settings  # noqa: E402
from features.batch import batch_id_to_date, date_to_batch_id  # noqa: E402
from model.io import save_booster  # noqa: E402
from model.train import (  # noqa: E402
    PARAMS,
    clean_labels,
    evaluate,
    fit,
    prepare_X,
    time_split,
)
from utils.log import setup_logger  # noqa: E402

logger = logging.getLogger(__name__)


def train(env, end_batch_id: Optional[int] = None) -> str:
    """Trains on the TRAIN_WINDOW_DAYS batches ending at end_batch_id (default:
    the latest), then registers + promotes (uc) or saves locally (local).
    Returns the model version ("local" in Stage 1)."""
    if end_batch_id is None:
        end_batch_id = connect.read_latest_batch_id(env, settings.FEATURES_TABLE)
    if end_batch_id is None:
        raise ValueError(
            f"{settings.FEATURES_TABLE} is empty. Run the feature_engineering "
            "backfill first (`invoke backfill --start ... --end ...`)."
        )
    start_date = batch_id_to_date(end_batch_id) - timedelta(
        days=settings.TRAIN_WINDOW_DAYS - 1
    )
    df, provenance = connect.read_features(
        env, date_to_batch_id(start_date), end_batch_id
    )
    df = clean_labels(df)
    if df.empty:
        raise ValueError(f"No usable feature rows for {provenance}.")

    train_df, test_df = time_split(df, settings.TEST_BATCHES)
    booster = fit(train_df, test_df)
    metrics = evaluate(booster, train_df, test_df)
    logger.info("Metrics: %s", metrics)

    if env.data_source == "local":
        save_booster(booster, settings.LOCAL_MODEL_PATH)
        logger.info("Saved local model to %s", settings.LOCAL_MODEL_PATH)
        return "local"
    return mlflow_utils.log_and_register(
        env, booster, prepare_X(test_df), PARAMS, metrics, provenance
    )


if __name__ == "__main__":
    parser = settings.base_arg_parser()
    parser.add_argument("--end-batch-id", type=settings.optional_int)
    args, _ = parser.parse_known_args()

    setup_logger(process_name="train")
    train(settings.env_from_args(args), args.end_batch_id)
