# Run from ml/: python -m scripts.export_source
"""Copies the whole source table to ml/data/01_raw/source_trips.parquet: the
"extract you were given" that Stage 0 (data_source=local) runs on."""

import connect
import settings


def main():
    df = connect.to_pandas(connect.get_spark().table(settings.SOURCE_TABLE))
    settings.LOCAL_SOURCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(settings.LOCAL_SOURCE_PATH, index=False)
    print(f"Wrote {len(df)} rows to {settings.LOCAL_SOURCE_PATH}")


if __name__ == "__main__":
    main()
