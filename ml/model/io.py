"""Local model files, for Stage 1 (data_source=local) only."""

from pathlib import Path

import lightgbm as lgb


def save_booster(booster: lgb.Booster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))


def load_booster(path: Path) -> lgb.Booster:
    if not path.exists():
        raise FileNotFoundError(
            f"No local model at {path}. Run "
            "`python -m pipelines.train --data-source local` first."
        )
    return lgb.Booster(model_file=str(path))
