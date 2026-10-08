# Run from ml/: python -m scripts.smoke_test_connect
import connect
import settings


def main():
    spark = connect.get_spark()
    spark.range(3).show()
    print(connect.to_pandas(spark.table(settings.SOURCE_TABLE), limit=5))


if __name__ == "__main__":
    main()
