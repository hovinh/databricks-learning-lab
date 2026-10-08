"""SQL release runner.

Runs the SQL scripts listed in a release manifest, in order, against one
catalog + schema, and records every statement in an audit table, so a failed
release can be rerun and resumes where it stopped.

Locally (Databricks Connect), from ml/:
    python ../sql/runner/run_sql_release.py --release-file releases/<file>.yaml \
        --sql-root ../sql --catalog workspace --schema taxi_dev --dry-run true
On Databricks: the `sql_release` job (resources/job_sql_release.yml).

A statement's identity is (script path, statement number in the script, hash of
the whitespace-normalised SQL), scoped to release name + catalog + schema. A
statement that already succeeded under that identity is skipped; editing it
changes the hash, so it runs again.
"""

import argparse
import hashlib
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml

AUDIT_TABLE_NAME = "log_sql_release"
AUDIT_SCHEMA = (
    "release_name STRING, catalog STRING, schema_name STRING, script_path STRING, "
    "statement_no INT, object_name STRING, statement_hash STRING, status STRING, "
    "error STRING, start_time TIMESTAMP, end_time TIMESTAMP, "
    "run_started_at TIMESTAMP"
)

# First match wins; only used to label statements in logs and the audit table.
OBJECT_PATTERNS = [
    r"CREATE\s+SCHEMA\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s;]+)",
    r"CREATE\s+(?:OR\s+REPLACE\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)",
    r"CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s;(]+)",
    r"CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s(]+)",
    r"ALTER\s+(?:TABLE|VIEW|SCHEMA)\s+([^\s(]+)",
    r"COMMENT\s+ON\s+(?:TABLE|SCHEMA|COLUMN)\s+([^\s]+)",
    r"DROP\s+(?:TABLE|VIEW|FUNCTION|SCHEMA)\s+(?:IF\s+EXISTS\s+)?([^\s;(]+)",
    r"INSERT\s+(?:INTO|OVERWRITE)\s+(?:TABLE\s+)?([^\s(]+)",
    r"MERGE\s+INTO\s+([^\s(]+)",
    r"DELETE\s+FROM\s+([^\s;]+)",
    r"(?:GRANT|REVOKE)\s+.*?\s+ON\s+(?:TABLE|VIEW|SCHEMA|CATALOG)?\s*([^\s;]+)",
    r"UPDATE\s+([^\s]+)",
]


# --- pure helpers (unit tested) ------------------------------------------------


def has_code(statement: str) -> bool:
    return any(
        line.strip() and not line.strip().startswith("--")
        for line in statement.splitlines()
    )


def split_statements(sql_text: str) -> list:
    """Splits on lines that END with ';'. One rule for authors: never end a line
    with ';' inside a string literal or a comment."""
    statements, buffer = [], []
    for line in sql_text.splitlines():
        buffer.append(line)
        if line.rstrip().endswith(";"):
            statement = "\n".join(buffer).strip().rstrip(";").strip()
            if has_code(statement):
                statements.append(statement)
            buffer = []
    tail = "\n".join(buffer).strip()
    if has_code(tail):
        statements.append(tail)
    return statements


def render(sql_text: str, catalog: str, schema: str) -> str:
    return sql_text.replace("{catalog}", catalog).replace("{schema}", schema)


def hash_statement(statement: str) -> str:
    normalised = " ".join(statement.split())
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]


def extract_object_name(statement: str) -> str:
    for pattern in OBJECT_PATTERNS:
        match = re.search(pattern, statement, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1)
    return "UNKNOWN"


def truncate_error(message: str, limit: int = 2000) -> str:
    cut = message.find("JVM stacktrace:")
    if cut != -1:
        message = message[:cut]
    return message.strip()[:limit]


def plan_release(scripts, sql_root: Path, catalog: str, schema: str, history: dict):
    """One record per statement, with action SKIP (succeeded before), RETRY (seen
    but not succeeded) or NEW (never seen)."""
    records = []
    for script_path in scripts:
        raw = (sql_root / script_path).read_text(encoding="utf-8")
        text = render(raw, catalog, schema)
        for statement_no, statement in enumerate(split_statements(text), start=1):
            statement_hash = hash_statement(statement)
            prior = history.get((script_path, statement_no, statement_hash))
            if prior == "SUCCESS":
                action = "SKIP"
            else:
                action = "RETRY" if prior else "NEW"
            records.append(
                {
                    "script_path": script_path,
                    "statement_no": statement_no,
                    "statement": statement,
                    "statement_hash": statement_hash,
                    "object_name": extract_object_name(statement),
                    "action": action,
                    "status": "NEW",
                    "error": None,
                    "start_time": None,
                    "end_time": None,
                }
            )
    return records


# --- Spark side ------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


def get_spark(profile: str):
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        from pyspark.sql import SparkSession

        return SparkSession.builder.getOrCreate()
    from databricks.connect import DatabricksSession

    return DatabricksSession.builder.serverless().profile(profile).getOrCreate()


def load_history(spark, audit_table, release_name, catalog, schema) -> dict:
    try:
        if not spark.catalog.tableExists(audit_table):
            return {}
    except Exception:  # schema doesn't exist yet
        return {}
    rows = spark.sql(
        f"SELECT script_path, statement_no, statement_hash, status FROM {audit_table} "
        "WHERE release_name = :release AND catalog = :catalog "
        "AND schema_name = :schema",
        args={"release": release_name, "catalog": catalog, "schema": schema},
    ).collect()
    return {
        (r["script_path"], r["statement_no"], r["statement_hash"]): r["status"]
        for r in rows
    }


def ensure_audit_table(spark, audit_table, catalog, schema) -> None:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    spark.sql(f"CREATE TABLE IF NOT EXISTS {audit_table} ({AUDIT_SCHEMA})")


def _audit_row(record, scope):
    release_name, catalog, schema, run_started_at = scope
    return (
        release_name,
        catalog,
        schema,
        record["script_path"],
        record["statement_no"],
        record["object_name"],
        record["statement_hash"],
        record["status"],
        record["error"],
        record["start_time"],
        record["end_time"],
        run_started_at,
    )


def prelog_new(spark, audit_table, records, scope) -> None:
    """Logs never-seen statements as NEW before anything runs, so the audit
    table shows the full plan even if this process dies mid-release."""
    rows = [_audit_row(r, scope) for r in records if r["action"] == "NEW"]
    if rows:
        new_rows = spark.createDataFrame(rows, schema=AUDIT_SCHEMA)
        new_rows.write.mode("append").saveAsTable(audit_table)


def execute(spark, records) -> bool:
    """Runs non-skipped statements in order, stopping at the first failure.
    Statements after a failure keep status NEW, so the next run resumes there."""
    for r in records:
        if r["action"] == "SKIP":
            continue
        r["start_time"] = _now()
        print(f"\n[{r['script_path']} #{r['statement_no']}] {r['object_name']}")
        print("    " + "\n    ".join(r["statement"].splitlines()))
        try:
            spark.sql(r["statement"])
            r["status"] = "SUCCESS"
        except Exception as exc:
            r["status"] = "FAILED"
            r["error"] = truncate_error(str(exc))
        r["end_time"] = _now()
        print(f"    -> {r['status']}" + (f": {r['error']}" if r["error"] else ""))
        if r["status"] == "FAILED":
            return False
    return True


def write_back(spark, audit_table, records, scope) -> None:
    rows = [_audit_row(r, scope) for r in records if r["action"] != "SKIP"]
    if not rows:
        return
    spark.createDataFrame(rows, schema=AUDIT_SCHEMA).createOrReplaceTempView(
        "sql_release_updates"
    )
    spark.sql(f"""
        MERGE INTO {audit_table} AS t
        USING sql_release_updates AS s
        ON  t.release_name = s.release_name AND t.catalog = s.catalog
        AND t.schema_name = s.schema_name AND t.script_path = s.script_path
        AND t.statement_no = s.statement_no AND t.statement_hash = s.statement_hash
        WHEN MATCHED THEN UPDATE SET
            t.object_name = s.object_name, t.status = s.status, t.error = s.error,
            t.start_time = s.start_time, t.end_time = s.end_time,
            t.run_started_at = s.run_started_at
        """)


# --- CLI -------------------------------------------------------------------------


def print_plan(records) -> None:
    for r in records:
        where = f"{r['script_path']} #{r['statement_no']}"
        print(f"  [{r['action']:<5}] {where}  {r['object_name']}")
    actions = [r["action"] for r in records]
    counts = {a: actions.count(a) for a in ("NEW", "RETRY", "SKIP")}
    print(f"Plan: {counts['NEW']} new, {counts['RETRY']} retry, {counts['SKIP']} skip")


def print_summary(records) -> None:
    ran = [r["status"] for r in records if r["action"] != "SKIP"]
    skipped = sum(r["action"] == "SKIP" for r in records)
    print(
        f"\nSucceeded: {ran.count('SUCCESS')}, failed: {ran.count('FAILED')}, "
        f"not reached: {ran.count('NEW')}, skipped: {skipped}"
    )


def _to_bool(value) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a SQL release manifest.")
    parser.add_argument("--release-file", required=True, help="Relative to --sql-root")
    parser.add_argument("--sql-root", required=True, help="Root of script paths")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--dry-run", type=_to_bool, default=False)
    parser.add_argument(
        "--profile", default=os.environ.get("DATABRICKS_CONFIG_PROFILE", "DEFAULT")
    )
    args, _ = parser.parse_known_args(argv)
    if not args.release_file:
        parser.error("--release-file is empty (set the job's release_file parameter)")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    # Connect logs every failed statement with its full JVM stack trace; the
    # runner already prints (and audits) a truncated error.
    for name in ("SQLQueryContextLogger", "pyspark.sql.connect.logging"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    sql_root = Path(args.sql_root)
    manifest_text = (sql_root / args.release_file).read_text(encoding="utf-8")
    manifest = yaml.safe_load(manifest_text)
    release_name, scripts = manifest["release"], manifest["scripts"]
    audit_table = f"{args.catalog}.{args.schema}.{AUDIT_TABLE_NAME}"
    scope = (release_name, args.catalog, args.schema, _now())

    target = f"{args.catalog}.{args.schema}"
    print(f"Release {release_name} -> {target} ({len(scripts)} scripts)")
    spark = get_spark(args.profile)
    history = load_history(spark, audit_table, release_name, args.catalog, args.schema)
    records = plan_release(scripts, sql_root, args.catalog, args.schema, history)
    print_plan(records)

    if args.dry_run:
        print("Dry run: nothing executed, nothing logged.")
        return
    if all(r["action"] == "SKIP" for r in records):
        print("Nothing to do: every statement already succeeded.")
        return

    ensure_audit_table(spark, audit_table, args.catalog, args.schema)
    prelog_new(spark, audit_table, records, scope)
    succeeded = execute(spark, records)
    write_back(spark, audit_table, records, scope)
    print_summary(records)
    if not succeeded:
        raise RuntimeError(
            f"Release {release_name} failed. Fix the cause and rerun the same "
            "release to resume from the failed statement."
        )


if __name__ == "__main__":
    main()
