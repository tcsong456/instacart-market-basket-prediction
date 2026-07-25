import pytest
from pyspark.sql.types import LongType, StructField, StructType

from instacart_etl.validation.unique import validate_uniqueness


def test_validate_uniqueness_passes_when_values_are_unique(spark):
    df = spark.createDataFrame(
        [
            (1, "prior"),
            (2, "prior"),
            (3, "train"),
        ],
        ["order_id", "eval_set"],
    )

    result = validate_uniqueness(
        df,
        columns=["order_id"],
    )

    assert result.passed is True
    assert result.failed_count == 0
    assert result.invalid_rows is None
    assert result.metadata["columns"] == ["order_id"]
    assert result.metadata["duplicate_values_count"] == 0
    assert result.metadata["duplicate_rows_count"] == 0
    assert result.message == "Columns 'order_id' have no duplicate values"
    assert result.rule_name == "order_id.uniqueness"
    assert result.category == "uniqueness"


def test_validate_uniqueness_detects_duplicate_values(spark):
    df = spark.createDataFrame(
        [
            (1, "prior"),
            (2, "prior"),
            (2, "train"),
            (3, "prior"),
            (3, "train"),
            (3, "test"),
        ],
        ["order_id", "eval_set"],
    )

    result = validate_uniqueness(
        df,
        columns=["order_id"],
    )

    assert result.passed is False
    assert result.failed_count == 5
    assert result.metadata["duplicate_values_count"] == 2
    assert result.metadata["duplicate_rows_count"] == 5
    assert result.message == "Columns 'order_id' found 2 duplicate key value(s)"


def test_validate_uniqueness_returns_duplicate_row_samples(spark):
    df = spark.createDataFrame(
        [(1, "a"), (2, "b"), (2, "c"), (3, "d"), (3, "e"), (3, "f")],
        ["order_id", "eval_set"],
    )

    result = validate_uniqueness(df, columns=["order_id"])

    invalid_rows = result.invalid_rows.collect()
    invalid_keys = {row["order_id"] for row in invalid_rows}

    assert invalid_keys == {2, 3}
    assert result.failed_count == 5


def test_validate_uniqueness_invalid_rows_limit(spark):
    df = spark.createDataFrame([(1,) for _ in range(50)], ["order_id"])

    result = validate_uniqueness(df, columns=["order_id"])

    assert result.invalid_rows is not None
    assert result.invalid_rows.count() == 30
    assert result.failed_count == 50


def test_validate_uniqueness_supports_unique_composite_key(spark):
    df = spark.createDataFrame(
        [
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
        ],
        ["user_id", "order_number"],
    )

    result = validate_uniqueness(
        df,
        columns=["user_id", "order_number"],
    )

    assert result.passed is True
    assert result.failed_count == 0
    assert result.invalid_rows is None
    assert result.rule_name == "user_id, order_number.uniqueness"
    assert result.message == "Columns 'user_id, order_number' have no duplicate values"


def test_validate_uniqueness_detects_duplicate_composite_keys(spark):
    df = spark.createDataFrame(
        [
            (1, 1, "a"),
            (1, 1, "b"),
            (1, 2, "c"),
            (2, 1, "d"),
        ],
        ["user_id", "order_number", "value"],
    )

    result = validate_uniqueness(
        df,
        columns=["user_id", "order_number"],
    )

    assert result.passed is False
    assert result.failed_count == 2
    assert result.metadata["duplicate_values_count"] == 1
    assert result.metadata["duplicate_rows_count"] == 2
    assert (
        result.message
        == "Columns 'user_id, order_number' found 1 duplicate key value(s)"
    )

    invalid_keys = {
        (row["user_id"], row["order_number"]) for row in result.invalid_rows.collect()
    }
    assert invalid_keys == {(1, 1)}


def test_validate_uniqueness_passes_empty_dataframe(spark):
    schema = StructType(
        [
            StructField(
                "order_id",
                LongType(),
                nullable=True,
            )
        ]
    )

    df = spark.createDataFrame([], schema)

    result = validate_uniqueness(
        df,
        columns=["order_id"],
    )

    assert result.passed is True
    assert result.failed_count == 0
    assert result.metadata["duplicate_values_count"] == 0
    assert result.metadata["duplicate_rows_count"] == 0


def test_validate_uniqueness_rejects_empty_column_names(spark):
    df = spark.createDataFrame(
        [(1,)],
        ["order_id"],
    )

    with pytest.raises(
        ValueError,
        match="column names cannot be empty",
    ):
        validate_uniqueness(
            df,
            columns=[],
        )


def test_validate_uniqueness_rejects_duplicate_column_names(spark):
    df = spark.createDataFrame(
        [(1, 10)],
        ["order_id", "user_id"],
    )

    with pytest.raises(
        ValueError,
        match="input column keys must be unique",
    ):
        validate_uniqueness(
            df,
            columns=["order_id", "order_id", "user_id"],
        )


def test_validate_uniqueness_ignores_nulls(spark):
    df = spark.createDataFrame(
        [
            (1,),
            (None,),
            (None,),
        ],
        ["order_id"],
    )

    result = validate_uniqueness(
        df,
        columns=["order_id"],
    )

    assert result.passed is True
    assert result.failed_count == 0
    assert result.metadata["duplicate_values_count"] == 0
    assert result.metadata["duplicate_rows_count"] == 0
