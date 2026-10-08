"""The only module that calls mlflow.*."""

import logging

import lightgbm as lgb
import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

import runtime
import settings

logger = logging.getLogger(__name__)


def configure_mlflow(env) -> None:
    """Inside Databricks, the bare URIs use the job's own identity. Locally
    there's no such context, so the CLI profile must be named in the URI."""
    if runtime.is_databricks():
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")
    else:
        profile = settings.DATABRICKS_PROFILE
        mlflow.set_tracking_uri(f"databricks://{profile}")
        mlflow.set_registry_uri(f"databricks-uc://{profile}")
    mlflow.set_experiment(env.experiment_path)


def log_and_register(env, booster, X_test, params, metrics, provenance) -> str:
    """Logs a training run, registers the model in UC and points the champion
    alias at the new version (unconditional promotion). Returns the version."""
    configure_mlflow(env)
    with mlflow.start_run(run_name="taxi_fare_train"):
        mlflow.log_params(params)
        mlflow.log_params({f"data_{k}": v for k, v in provenance.items()})
        mlflow.log_params(
            {
                "catalog": env.catalog,
                "schema": env.schema,
                "best_iteration": booster.best_iteration,
            }
        )
        mlflow.log_metrics(metrics)

        # UC requires a signature. X_test is already all-float64 (prepare_X),
        # so the signature never hard-codes "non-nullable integer" columns.
        signature = infer_signature(X_test, booster.predict(X_test))
        model_info = mlflow.lightgbm.log_model(
            booster,
            name="model",
            signature=signature,
            input_example=X_test.head(5),
            # Only what inference needs, from the versions actually installed,
            # so the model never claims a dependency the predict job lacks.
            pip_requirements=[
                f"lightgbm=={lgb.__version__}",
                f"pandas=={pd.__version__}",
            ],
        )

    version = mlflow.register_model(model_info.model_uri, env.model_name).version
    MlflowClient().set_registered_model_alias(
        env.model_name, settings.CHAMPION_ALIAS, version
    )
    logger.info(
        "Registered %s v%s as @%s", env.model_name, version, settings.CHAMPION_ALIAS
    )
    return str(version)


def load_champion(env):
    """Returns (pyfunc model, version). The alias is resolved to a concrete
    version first, so the version recorded with predictions is the one used."""
    configure_mlflow(env)
    try:
        version = (
            MlflowClient()
            .get_model_version_by_alias(env.model_name, settings.CHAMPION_ALIAS)
            .version
        )
    except MlflowException as exc:
        raise RuntimeError(
            f"No @{settings.CHAMPION_ALIAS} version of {env.model_name}. Run the "
            "retrain job (or `python -m pipelines.train`) first."
        ) from exc
    model = mlflow.pyfunc.load_model(f"models:/{env.model_name}/{version}")
    logger.info("Loaded %s v%s", env.model_name, version)
    return model, str(version)
