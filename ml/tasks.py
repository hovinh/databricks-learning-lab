from invoke import task

LINT_TARGETS = (
    "settings.py runtime.py connect.py mlflow_utils.py features model pipelines "
    "utils scripts tests tasks.py ../sql/runner"
)


@task
def test(c):
    c.run("pytest")


@task
def lint(c):
    c.run(f"flake8 {LINT_TARGETS}")


@task
def format(c):
    c.run(f"black {LINT_TARGETS} && isort {LINT_TARGETS}")


@task(pre=[format, lint, test])
def all(c):
    print("All checks passed.")


@task
def run_local(c, data_source="uc"):
    """feature_engineering -> train -> predict, locally. data_source=local needs
    no Databricks at all (run scripts.export_source once first)."""
    for pipeline in ("feature_engineering", "train", "predict"):
        c.run(f"python -m pipelines.{pipeline} --data-source {data_source}")


@task
def backfill(c, start, end, data_source="uc"):
    """Builds feature batches for every day from start to end (YYYY-MM-DD)."""
    c.run(
        "python -m pipelines.feature_engineering "
        f"--start-date {start} --end-date {end} --data-source {data_source}"
    )


@task
def sql_release(c, release, schema="taxi_dev", catalog="workspace", dry_run=False):
    """Runs a SQL release manifest locally through Databricks Connect."""
    c.run(
        "python ../sql/runner/run_sql_release.py "
        f"--release-file {release} --sql-root ../sql "
        f"--catalog {catalog} --schema {schema} --dry-run {str(dry_run).lower()}"
    )
