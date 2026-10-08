# CLAUDE.md

Demo repo for a blog post: an NYC taxi fare model (LightGBM) on Databricks Free Edition.
Plain Python developed locally, run against Unity Catalog via Databricks Connect, deployed as
serverless jobs with a Databricks Asset Bundle. Table DDL goes through a SQL release runner.

## Commands

Python commands run from `ml/` with the venv active (Python **3.12**, matches serverless):

```bash
invoke all                                           # black + isort -> flake8 -> pytest: definition of done
invoke run-local [--data-source local]               # feature_engineering -> train -> predict
invoke backfill --start 2016-01-08 --end 2016-02-15 [--data-source local]
invoke sql-release --release releases/<file>.yaml [--schema taxi_sandbox] [--dry-run]
python -m pipelines.<name> [--schema taxi_prd]       # one pipeline; always -m, never the file path
python -m scripts.smoke_test_connect                 # Connect sanity check
```

Bundle commands run from the repo root: `databricks bundle validate|deploy|run <job>|summary [-t prd]`.
Jobs: `taxi_predict` (feature_engineering -> predict), `taxi_retrain`, `sql_release`
(`--params release_file=releases/<file>.yaml`).

Dependencies: edit `ml/requirements.in` (pinned `==`), then
`python -m piptools compile requirements.in -o requirements.txt` and `python -m piptools sync requirements.txt`.
Never `pip install` into the venv directly. Keep pandas/numpy/lightgbm/mlflow pins in sync with the
`ml_dependencies` variable in `databricks.yml`.

## Architecture rules

- `ml/connect.py` is the only module touching Spark/UC; `ml/mlflow_utils.py` the only one calling `mlflow.*`.
  `features/` and `model/` are pure pandas: no I/O, no logging setup.
- Every function takes an explicit `settings.Env` (catalog, schema, data_source); no import-time globals.
- Two switches: `runtime.is_databricks()` (where code runs) and `--data-source uc|local` (where data and model live).
- Pipelines (`ml/pipelines/*.py`) keep the `__file__`/`filename` sys.path bootstrap at the top (needed by
  serverless `spark_python_task`) and call `setup_logger()` only under `__main__`.
- Writes go through `connect.write_batch` (idempotent `replaceWhere` per `batch_id`, columns aligned to the table schema).
- Fail loudly: raise with a hint instead of logging and returning.
- Table columns live in `sql/tables/*.sql` AND `settings.*_TABLE_COLUMNS`; `tests/test_ddl_contract.py` enforces they match.
- SQL objects change only via a new release manifest in `sql/releases/` with a new `release:` name.
  Tables: `CREATE TABLE IF NOT EXISTS` + additive `ALTER`s, never `CREATE OR REPLACE TABLE`. Keep `;` only at statement ends.
- Tests run offline: mock at the `connect.*` / `mlflow_utils.*` boundary; `conftest.py` fails any test that opens Spark.

## Environments

Workspace catalog `workspace`; schema `taxi_dev` (target `dev`, default) and `taxi_prd` (target `prd`).
`prd` schedules deploy paused (`presets.trigger_pause_status`). The source `samples.nyctaxi.trips` covers
2016-01-01..2016-02-29; each feature run advances a simulated clock by one day.

## Gotchas

- Databricks Connect / MLflow get tokens by calling the `databricks` CLI (`auth_type = databricks-cli`):
  it must be on the PATH of the Python process.
- Windows: MLflow prints emoji; `setup_logger` makes stdout tolerate cp1252 pipes.
- Use `toPandas()`/SQL for timestamps; `collect()` converts them to the laptop's time zone.
- Do not install `pyspark` next to `databricks-connect`.
- Never commit workspace URLs, IDs or emails (the repo is published with the blog).
