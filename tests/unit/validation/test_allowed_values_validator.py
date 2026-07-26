import pytest
from pyspark.sql.types import StringType, StructField, StructType

from instacart_etl.validation.allowed_values import validate_allowed_values


def test_validate_allowed_values_passes_valid_values(spark):
    df = spark.createDataFrame(
        [
            ("prior",),
            ("train",),
            ("test",),
        ],
        ["eval_set"],
    )

    result = validate_allowed_values(
        df, column_name="eval_set", allowed_values=["prior", "train", "test"]
    )

    assert result.rule_name == "eval_set.allowed_values"
    assert result.category == "allowed_values"
    assert result.passed is True
    assert result.failed_count == 0
    assert result.invalid_rows is None
    assert result.message == "Column: all values of 'eval_set' are allowed"


def test_validate_allowed_values_detects_invalid_values(spark):
    df = spark.createDataFrame(
        [("prior",), ("invalid",), ("train",), ("other",), ("other",)],
        ["eval_set"],
    )

    result = validate_allowed_values(
        df, column_name="eval_set", allowed_values=["prior", "train", "test"]
    )

    assert result.passed is False
    assert result.failed_count == 3
    assert result.message == (
        "Column 'eval_set' contains 3 row(s) with disallowed values"
    )

    invalid_rows = sorted([row["eval_set"] for row in result.invalid_rows.collect()])
    assert invalid_rows == ["invalid", "other"]


def test_validate_allowed_values_ignores_nulls(spark):
    df = spark.createDataFrame(
        [("train",), ("test",), (None,), ("prior",), (None,)], ["eval_set"]
    )

    result = validate_allowed_values(
        df, column_name="eval_set", allowed_values=["prior", "train", "test"]
    )

    assert result.passed is True
    assert result.failed_count == 0
    assert result.invalid_rows is None


def test_validate_allowed_values_empty_dataframe(spark):
    schema = StructType([StructField("eval_set", StringType(), True)])
    df = spark.createDataFrame([], schema=schema)

    result = validate_allowed_values(
        df, column_name="eval_set", allowed_values=["prior", "train", "test"]
    )

    assert result.passed is True
    assert result.failed_count == 0


def test_validate_allowed_values_empty_allowed_values_list(spark):
    df = spark.createDataFrame([("prior",)], ["eval_set"])

    with pytest.raises(ValueError, match="allowed_values can not be empty"):
        validate_allowed_values(df, column_name="eval_set", allowed_values=[])


def test_validate_allowed_values_invalid_rows_limit(spark):
    df = spark.createDataFrame([(value,) for value in range(40)], ["value"])

    result = validate_allowed_values(df, column_name="value", allowed_values=[40])

    invalid_rows = [row["value"] for row in result.invalid_rows.collect()]

    assert result.failed_count == 40
    assert result.invalid_rows.count() == 20
    assert len(set(invalid_rows)) == 20
    assert set(invalid_rows).issubset(set(range(40)))
