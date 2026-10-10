# databricks-learning-lab

A small, complete ML project on **Databricks Free Edition**: a LightGBM model that predicts NYC taxi
fares from the built-in `samples.nyctaxi.trips` table, taken from a laptop prototype to scheduled
serverless jobs in three stages.

| Stage | Code runs on | Data and model live in | How |
|---|---|---|---|
| 1. Fully local | laptop | parquet files and a model file under `ml/data/` | `--data-source local` |
| 2. Hybrid | laptop | Unity Catalog tables and model registry, through Databricks Connect | `--data-source uc` (default) |
| 3. Remote | serverless jobs | Unity Catalog | `databricks bundle deploy` / `run` |

The same files run in all three stages. The walkthrough is in the companion blog post
<!-- TODO(migrate): link the published post -->
(draft in [`blog/databricks-ml-setup/`](blog/databricks-ml-setup/)).

## What's inside

- Three pipelines: feature engineering, training and prediction (`ml/pipelines/`).
- The project's own Unity Catalog schema, tables and view, created by a resumable SQL release
  runner (`sql/`).
- MLflow tracking and the Unity Catalog model registry, with a `@champion` alias.
- Three jobs, defined with a Databricks Asset Bundle: `taxi_predict` (daily), `taxi_retrain`
  (weekly) and `sql_release` (manual).
- Two environments on one workspace: `dev` (schema `workspace.taxi_dev`) and `prd`
  (`workspace.taxi_prd`).
- An offline test suite that never opens a Spark session.

## Working config

This is the combination the demo was built and run with (October 2026). Other versions may work,
but the Python versions have to agree.

| Thing | Version |
|---|---|
| Databricks workspace | Free Edition (serverless only) |
| Databricks CLI | v1.20.0 or newer |
| Local Python | 3.12 (must match the serverless runtime) |
| `databricks-connect` | 17.3.12 |
| Serverless `environment_version` for jobs | `"4"` (Python 3.12.3, pairs with Connect 17.3) |
| pandas / numpy / lightgbm / mlflow | 2.3.3 / 2.5.1 / 4.7.0 / 3.15.1 |

All direct dependencies are pinned in [`ml/requirements.in`](ml/requirements.in), and the job
libraries in the `ml_dependencies` variable of [`databricks.yml`](databricks.yml). Keep the two in
sync.

## Quickstart

```bash
# 1. Databricks CLI: winget install Databricks.DatabricksCLI   (or brew install databricks/tap/databricks)
databricks auth login --host https://<your-workspace-host> --profile DEFAULT
databricks current-user me --profile DEFAULT

# 2. Python 3.12 venv outside the repo, then the locked dependencies
py -3.12 -m venv ~/.venvs/taxi            # or: python3.12 -m venv ~/.venvs/taxi
source ~/.venvs/taxi/Scripts/activate     # macOS/Linux: ~/.venvs/taxi/bin/activate
cd ml
python -m pip install --upgrade pip pip-tools
python -m piptools sync requirements.txt

# 3. Is Databricks Connect working?
python -m scripts.smoke_test_connect
```

Python commands run from `ml/` with the venv active. Bundle commands run from the repo root.

## Running the three stages

**Stage 1, fully local.** Export the source table to parquet once (this one step needs Connect),
then nothing touches Databricks:

```bash
# from ml/
python -m scripts.export_source                                   # -> ml/data/01_raw/source_trips.parquet
invoke backfill --start 2016-01-08 --end 2016-02-15 --data-source local
invoke run-local --data-source local                              # features -> train -> predict
```

**Stage 2, hybrid.** Create the tables in your dev schema, then run the same commands without
`--data-source local`:

```bash
# from ml/
invoke sql-release --release releases/20261006_sql_release_init.yaml
invoke backfill --start 2016-01-08 --end 2016-02-15
invoke run-local
```

**Stage 3, remote.** Deploy the bundle and run the jobs. On a fresh schema the order is: SQL
release, backfill (Stage 2 above), retrain, predict.

```bash
# from the repo root
databricks bundle validate
databricks bundle deploy                                          # target dev (default)
databricks bundle run sql_release --params release_file=releases/20261006_sql_release_init.yaml
databricks bundle run taxi_retrain
databricks bundle run taxi_predict
databricks bundle summary

databricks bundle deploy -t prd                                   # prd schedules deploy paused
```

The source table covers 2016-01-01 to 2016-02-29. Each run of `feature_engineering` without a date
processes the next unprocessed day, so `taxi_predict` advances a simulated clock by one day per run
and fails with a clear message once the data runs out. That's why `prd` deploys its schedules
paused.

## Commands

```bash
# from ml/, venv active
invoke all                                             # black + isort -> flake8 -> pytest
invoke run-local [--data-source local]                 # feature_engineering -> train -> predict
invoke backfill --start 2016-01-08 --end 2016-02-15 [--data-source local]
invoke sql-release --release releases/<file>.yaml [--schema taxi_sandbox] [--dry-run]
python -m pipelines.<name> [--schema taxi_prd] [--data-source local]   # one pipeline; always -m
python -m scripts.smoke_test_connect

# from the repo root
databricks bundle validate | deploy | summary [-t prd]
databricks bundle run taxi_predict | taxi_retrain [-t prd]
databricks bundle run sql_release --params release_file=releases/<file>.yaml
```

To change dependencies, edit `ml/requirements.in`, then
`python -m piptools compile requirements.in -o requirements.txt` and
`python -m piptools sync requirements.txt`.

## Project layout

```text
databricks-learning-lab/
├── databricks.yml              # bundle root: variables, targets (dev, prd)
├── resources/                  # one YAML per job
├── sql/
│   ├── schemas/ tables/ views/ # DDL, one object per file, {catalog}/{schema} placeholders
│   ├── releases/               # release manifests: what to run, in which order
│   └── runner/                 # the SQL release runner (local or as the sql_release job)
└── ml/
    ├── connect.py              # the only module touching Spark / Unity Catalog
    ├── mlflow_utils.py         # the only module calling mlflow.*
    ├── features/  model/       # pure pandas logic, no I/O
    ├── pipelines/              # entry points the jobs run
    ├── settings.py             # names, features, Env, CLI arguments
    ├── runtime.py              # is_databricks(), task values, widgets
    ├── utils/  scripts/  tests/
    ├── tasks.py                # invoke tasks
    └── requirements.in, requirements.txt
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `cannot configure default credentials ... auth_type=databricks-cli` | Connect and MLflow get their token from the `databricks` CLI, which isn't on the Python process's `PATH` | restart the terminal or IDE after installing the CLI |
| `MlflowException: ... not in a valid JSON format` | the profile's `host` is a browser URL with a path | `databricks auth login` with the bare workspace host |
| `ModuleNotFoundError: No module named 'settings'` | a pipeline was run as a file from the wrong folder | `python -m pipelines.<name>` from `ml/` |
| `Could not find a version that satisfies the requirement` in a job | the job's `environment_version` runs an older Python than the library needs | bump `environment_version` |
| `bundle run` shows only a JVM stack trace | the Python error isn't streamed | `databricks jobs get-run-output <task_run_id>`, or the run page |
| `UnicodeEncodeError: 'charmap' codec ...` on Windows | MLflow prints emoji into a cp1252 pipe | already handled by `setup_logger`; or set `PYTHONUTF8=1` |
| `openpgp: key expired` on `bundle deploy` | an old CLI's embedded Terraform signing key expired | upgrade the CLI |
| pip fails halfway on Windows | a path longer than 260 characters | put the venv at a short path |
| `SSLCertVerificationError: self-signed certificate` | a corporate TLS-inspection proxy | `pip install pip-system-certs` |
| timestamps off by your UTC offset | `collect()` converts to the laptop's time zone | date logic in SQL or after `toPandas()` |
| `No @champion version of ...` | predict ran before any training | run `taxi_retrain` (or `python -m pipelines.train`) first |
| `Simulated clock reached the end of the source data` | every day of the sample has been processed | pass `--batch-date`, or drop the features table to restart |

Don't install `pyspark` next to `databricks-connect`: Connect ships its own.
