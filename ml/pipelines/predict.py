import sys
from pathlib import Path

try:
    _entry = Path(__file__)
except NameError:
    _entry = Path(filename)  # noqa: F821
sys.path.insert(0, str(_entry.resolve().parent.parent))

import logging  # noqa: E402
from typing import Optional  # noqa: E402

import numpy as np  # noqa: E402

import connect  # noqa: E402
import mlflow_utils  # noqa: E402
import settings  # noqa: E402
from features.batch import utc_now  # noqa: E402
from model.io import load_booster  # noqa: E402
from model.train import prepare_X  # noqa: E402
from utils.log import setup_logger  # noqa: E402

logger = logging.getLogger(__name__)

# Carried alongside the features and reattached after predicting: MLflow's
# signature enforcement drops every column the signature doesn't declare.
ID_COLUMNS = ["trip_id", "batch_id", "pickup_ts"]


def predict(env, batch_id: Optional[int] = None) -> int:
    """Scores one feature batch (default: the latest) and writes predictions.
    Returns the number of rows written."""
    source = "given"
    if batch_id is None:
        batch_id = connect.read_latest_batch_id(env, settings.FEATURES_TABLE)
        source = "latest in table"
    if batch_id is None:
        raise ValueError(f"{settings.FEATURES_TABLE} is empty: nothing to score.")
    logger.info("Scoring batch_id=%d (%s)", batch_id, source)

    df, _ = connect.read_features(env, batch_id, batch_id)
    if df.empty:
        raise ValueError(f"No feature rows for batch_id={batch_id}.")
    missing = [c for c in settings.FEATURES + ID_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Feature rows are missing columns: {missing}")

    X = prepare_X(df)
    if env.data_source == "local":
        predictions = load_booster(settings.LOCAL_MODEL_PATH).predict(X)
        version = "local"
    else:
        model, version = mlflow_utils.load_champion(env)
        predictions = model.predict(X)

    out = df[ID_COLUMNS].copy()
    out["predicted_fare"] = np.asarray(predictions, dtype="float64")
    out["model_version"] = version
    out["created_at"] = utc_now()
    connect.write_batch(
        env, out[settings.PREDICTION_TABLE_COLUMNS], settings.PREDICTIONS_TABLE
    )
    logger.info("Wrote %d predictions for batch_id=%d", len(out), batch_id)
    return len(out)


if __name__ == "__main__":
    parser = settings.base_arg_parser()
    parser.add_argument("--batch-id", type=settings.optional_int)
    args, _ = parser.parse_known_args()

    setup_logger(process_name="predict")
    predict(settings.env_from_args(args), args.batch_id)
