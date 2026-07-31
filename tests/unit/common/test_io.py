from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from instacart_etl_rnn.common.io import read_csv, read_parquet


def test_read_csv_with_schema(spark, tmp_path):
    csv_path = tmp_path / "test.csv"

    csv_path.write_text(
        "id,name\n5,Alice\n11,Bob\n29,Tony\n",
        encoding="utf-8",
    )

    schema = StructType(
        [
            StructField("id", IntegerType(), False),
            StructField("name", StringType(), False),
        ]
    )

    df = read_csv(
        spark=spark,
        path=csv_path,
        schema=schema,
    )

    assert df.columns == ["id", "name"]

    assert df.schema["id"].dataType == IntegerType()
    assert df.schema["name"].dataType == StringType()

    assert df.count() == 3

    assert [tuple(row) for row in df.orderBy("id").collect()] == [
        (5, "Alice"),
        (11, "Bob"),
        (29, "Tony"),
    ]


def test_read_csv_infers_schema(spark, tmp_path):
    csv_path = tmp_path / "test.csv"

    csv_path.write_text("id,name,price\n1,Alice,10.5\n2,Bob,20.0\n3,Tony,15.75\n")

    df = read_csv(
        path=csv_path,
        spark=spark,
    )

    assert df.count() == 3

    assert df.columns == [
        "id",
        "name",
        "price",
    ]

    assert df.schema["id"].dataType == IntegerType()
    assert df.schema["name"].dataType == StringType()
    assert df.schema["price"].dataType == DoubleType()

    assert [tuple(row) for row in df.orderBy("id").collect()] == [
        (1, "Alice", 10.5),
        (2, "Bob", 20.0),
        (3, "Tony", 15.75),
    ]


def test_read_parquet(spark, tmp_path):
    expected_df = spark.createDataFrame(
        [(5, "Alice"), (11, "Bob"), (29, "Tony")],
        ["id", "name"],
    )

    input_path = tmp_path / "parquet"
    expected_df.write.parquet(str(input_path))

    actual_df = read_parquet(
        path=input_path,
        spark=spark,
    )

    assert actual_df.schema == expected_df.schema
    assert actual_df.orderBy("id").collect() == expected_df.orderBy("id").collect()


def test_write_parquet(spark, tmp_path):
    df = spark.createDataFrame(
        [(5, "Alice"), (11, "Bob"), (29, "Tony")],
        ["id", "name"],
    )

    output_path = tmp_path / "parquet"

    df.write.parquet(str(output_path))

    loaded_df = read_parquet(
        path=output_path,
        spark=spark,
    )

    assert loaded_df.schema == df.schema
    assert loaded_df.orderBy("id").collect() == df.orderBy("id").collect()
