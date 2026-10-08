import run_sql_release as runner


def test_splits_on_line_ending_semicolons_and_drops_comment_only_chunks():
    sql = "CREATE TABLE a (x INT);\n-- note;\nCREATE VIEW v AS\nSELECT 1;\n"
    assert runner.split_statements(sql) == [
        "CREATE TABLE a (x INT)",
        "CREATE VIEW v AS\nSELECT 1",
    ]


def test_hash_ignores_whitespace_but_not_content():
    base = runner.hash_statement("CREATE TABLE a (x INT)")
    assert runner.hash_statement("CREATE   TABLE a\n  (x INT)") == base
    assert runner.hash_statement("CREATE TABLE a (y INT)") != base


def test_render_substitutes_placeholders():
    assert runner.render("{catalog}.{schema}.t", "c", "s") == "c.s.t"


def test_extract_object_name():
    stmt = "CREATE TABLE IF NOT EXISTS c.s.taxi_features (x INT)"
    assert runner.extract_object_name(stmt) == "c.s.taxi_features"


def test_plan_classifies_skip_retry_new(tmp_path):
    (tmp_path / "a.sql").write_text(
        "CREATE TABLE {catalog}.{schema}.t1 (x INT);\n"
        "CREATE TABLE {catalog}.{schema}.t2 (x INT);\n"
    )
    (tmp_path / "b.sql").write_text("CREATE VIEW {catalog}.{schema}.v AS SELECT 1;\n")
    h1 = runner.hash_statement("CREATE TABLE c.s.t1 (x INT)")
    h2 = runner.hash_statement("CREATE TABLE c.s.t2 (x INT)")
    history = {("a.sql", 1, h1): "SUCCESS", ("a.sql", 2, h2): "FAILED"}
    plan = runner.plan_release(["a.sql", "b.sql"], tmp_path, "c", "s", history)
    assert [r["action"] for r in plan] == ["SKIP", "RETRY", "NEW"]


def test_edited_statement_is_new(tmp_path):
    (tmp_path / "a.sql").write_text("CREATE TABLE c.s.t1 (x INT, y INT);\n")
    old_hash = runner.hash_statement("CREATE TABLE c.s.t1 (x INT)")
    history = {("a.sql", 1, old_hash): "SUCCESS"}
    plan = runner.plan_release(["a.sql"], tmp_path, "c", "s", history)
    assert plan[0]["action"] == "NEW"
