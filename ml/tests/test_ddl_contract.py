import re

import pytest

import settings

SQL_TABLES = settings.ML_ROOT.parent / "sql" / "tables"
COLUMN_LINE = re.compile(
    r"^\s*,?\s*([a-z_][a-z0-9_]*)\s+"
    r"(BIGINT|INT|DOUBLE|STRING|TIMESTAMP|DATE|BOOLEAN)\b",
    re.IGNORECASE,
)


def ddl_columns(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [m.group(1) for m in map(COLUMN_LINE.match, lines) if m]


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("taxi_features.sql", settings.FEATURE_TABLE_COLUMNS),
        ("taxi_predictions.sql", settings.PREDICTION_TABLE_COLUMNS),
    ],
)
def test_code_columns_match_ddl(filename, expected):
    assert ddl_columns(SQL_TABLES / filename) == expected
