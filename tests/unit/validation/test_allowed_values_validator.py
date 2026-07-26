import pytest
from pyspark.sql.types import StringType, StructField, StructType

from instacart_etl.validation.allowed_values import validate_allowed_values
from instacart_etl.validation.exceptions import InvalidConstraintError


def test_allowed_values_validator_passes_valid_values(spark):
    df = spark.createDataFrame(
        [("train", 1), ("test", 3), ("prior", 5)], ["eval_set", "category"]
    )

    contract = {
        "schema": [
            {
                "name": "eval_set",
                "constraints": {"allowed_values": ["test", "train", "prior"]},
            },
            {"name": "category", "constraints": {"allowed_values": [1, 2, 3, 4, 5]}},
        ]
    }

    results = validate_allowed_values(df, contract=contract)

    assert len(results) == 2
    assert all(result.passed for result in results)
    assert all(result.failed_count == 0 for result in results)
    assert all(result.invalid_rows is None for result in results)
    assert results[0].rule_name == "eval_set.allowed_values"
    assert results[1].rule_name == "category.allowed_values"
    assert all(result.category == "allowed_values" for result in results)
    assert results[0].message == "Column: all values of 'eval_set' are allowed"
    assert results[1].message == "Column: all values of 'category' are allowed"
    assert results[0].metadata == {
        "column_name": "eval_set",
        "allowed_values": ["test", "train", "prior"],
    }
    assert results[1].metadata == {
        "column_name": "category",
        "allowed_values": [1, 2, 3, 4, 5],
    }


def test_allowed_values_validator_detects_invalid_values(spark):
    df = spark.createDataFrame(
        [("train", 1), ("invalid", 3), ("prior", 5), ("other", 2)],
        ["eval_set", "category"],
    )

    contract = {
        "schema": [
            {
                "name": "eval_set",
                "constraints": {"allowed_values": ["test", "train", "prior"]},
            },
            {"name": "category", "constraints": {"allowed_values": [1, 2, 3, 4, 5]}},
        ]
    }

    results = validate_allowed_values(df, contract=contract)

    assert results[0].passed is False
    assert results[0].failed_count == 2
    assert results[0].message == (
        "Column 'eval_set' contains 2 row(s) with disallowed values"
    )

    invalid_rows = {row["eval_set"] for row in results[0].invalid_rows.collect()}
    assert invalid_rows == {"invalid", "other"}

    assert results[1].passed is True
    assert results[1].failed_count == 0
    assert results[1].invalid_rows is None


def test_allowed_values_validator_no_constraints(spark):
    df = spark.createDataFrame(
        [("train", 1), ("test", 3), ("prior", 5)], ["eval_set", "category"]
    )

    contract = {"schema": [{"name": "eval_set"}, {"name": "category"}]}

    results = validate_allowed_values(df, contract=contract)

    assert results == []


def test_allowed_values_validator_no_allowed_values_constraints(spark):
    df = spark.createDataFrame(
        [("train", 1), ("test", 3), ("prior", 5)], ["eval_set", "category"]
    )

    contract = {
        "schema": [
            {"name": "eval_set", "constraints": {"mode": "train"}},
            {"name": "category", "constraints": {"maximum": 10}},
        ]
    }

    results = validate_allowed_values(df, contract=contract)

    assert results == []


def test_allowed_values_validator_ignore_nulls(spark):
    df = spark.createDataFrame(
        [("train", 1), (None, 6), ("prior", None), (None, 2), ("test", 0)],
        ["eval_set", "category"],
    )

    contract = {
        "schema": [
            {
                "name": "eval_set",
                "constraints": {"allowed_values": ["test", "train", "prior"]},
            },
            {"name": "category", "constraints": {"allowed_values": [1, 2, 3, 4, 5]}},
        ]
    }

    results = validate_allowed_values(df, contract=contract)

    assert results[0].passed is True
    assert results[1].passed is False
    assert results[1].failed_count == 2

    invalid_rows = {row["category"] for row in results[1].invalid_rows.collect()}
    assert invalid_rows == {0, 6}


def test_allowed_values_validator_empty_allowed_values_list(spark):
    df = spark.createDataFrame([("prior",)], ["eval_set"])

    contract = {"schema": [{"name": "eval_set", "constraints": {"allowed_values": []}}]}

    with pytest.raises(
        InvalidConstraintError, match="allowed_values must not be an empty list"
    ):
        validate_allowed_values(df, contract=contract)


def test_allowed_values_validator_invalid_allowed_values(spark):
    df = spark.createDataFrame([("prior",)], ["eval_set"])

    contract = {
        "schema": [
            {
                "name": "eval_set",
                "constraints": {
                    "allowed_values": {
                        "allowed_value_1": "train",
                        "allowed_value_2": "test",
                    }
                },
            }
        ]
    }

    with pytest.raises(
        InvalidConstraintError, match="allowed_values in constraint must be a list"
    ):
        validate_allowed_values(df, contract=contract)


def test_validate_allowed_values_invalid_rows_limit(spark):
    df = spark.createDataFrame([(value,) for value in range(40)], ["value"])

    contract = {"schema": [{"name": "value", "constraints": {"allowed_values": [40]}}]}

    result = validate_allowed_values(df, contract=contract)[0]

    invalid_rows = [row["value"] for row in result.invalid_rows.collect()]

    assert result.failed_count == 40
    assert result.invalid_rows.count() == 20
    assert len(set(invalid_rows)) == 20
    assert set(invalid_rows).issubset(set(range(40)))


def test_validate_allowed_values_empty_dataframe(spark):
    schema = StructType([StructField("sales", StringType(), False)])
    df = spark.createDataFrame([], schema=schema)

    contract = {"schema": [{"name": "value", "constraints": {"allowed_values": [40]}}]}

    result = validate_allowed_values(df, contract=contract)[0]

    assert result.passed is True
    assert result.failed_count == 0
    assert result.invalid_rows is None


def test_validate_allowed_values_all_values_null(spark):
    df = spark.createDataFrame(
        [
            (None),
            (None),
        ]
    )

    contract = {
        "schema": [{"name": "eval_set", "constraints": {"allowed_values": ["train"]}}]
    }

    result = validate_allowed_values(df, contract=contract)[0]

    assert result.passed is True
    assert result.failed_count == 0
