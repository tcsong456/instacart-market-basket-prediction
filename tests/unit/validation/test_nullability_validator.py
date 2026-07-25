import pytest
from pyspark.sql.types import (
    IntegerType,
    StructField,
    StructType,
)

from instacart_etl.validation.exceptions import InvalidConstraintError
from instacart_etl.validation.null import validate_nullability


def test_nullability_validator_passes_non_nullable_columns(spark):
    schema = StructType(
        [
            StructField("order_id", IntegerType(), False),
            StructField("user_id", IntegerType(), False),
        ]
    )
    df = spark.createDataFrame([(1, 10), (2, 20)], schema=schema)

    contract = {
        "schema": [
            {"name": "order_id", "type": "integer", "nullable": False},
            {"name": "user_id", "type": "integer", "nullable": False},
        ]
    }

    results = validate_nullability(df, contract=contract)

    assert len(results) == 2
    assert all(result.passed for result in results)
    assert all(result.failed_count == 0 for result in results)
    assert all(result.invalid_rows is None for result in results)
    assert results[0].rule_name == "order_id.nullability"
    assert results[1].rule_name == "user_id.nullability"
    assert all(result.category == "nullability" for result in results)
    assert results[0].message == "Column 'order_id' contains no null values"
    assert results[1].message == "Column 'user_id' contains no null values"
    assert results[0].metadata == {"column_name": "order_id", "nullable": False}
    assert results[1].metadata == {"column_name": "user_id", "nullable": False}


def test_nullability_validator_detects_nulls(spark):
    schema = StructType(
        [
            StructField("order_id", IntegerType(), True),
            StructField("user_id", IntegerType(), True),
        ]
    )
    df = spark.createDataFrame([(1, 10), (None, 20), (None, None)], schema=schema)

    contract = {
        "schema": [
            {
                "name": "order_id",
                "type": "integer",
                "nullable": False,
            },
            {
                "name": "user_id",
                "type": "integer",
                "nullable": False,
            },
        ]
    }

    results = validate_nullability(df, contract=contract)
    results_by_column = {result.metadata["column_name"]: result for result in results}

    assert results_by_column["order_id"].passed is False
    assert results_by_column["order_id"].failed_count == 2
    assert results_by_column["order_id"].invalid_rows is not None
    assert results_by_column["order_id"].invalid_rows.count() == 2
    assert (
        results_by_column["order_id"].message
        == "Column 'order_id' is not nullable but contains 2 null value(s)"
    )

    assert results_by_column["user_id"].passed is False
    assert results_by_column["user_id"].failed_count == 1
    assert results_by_column["user_id"].invalid_rows is not None
    assert results_by_column["user_id"].invalid_rows.count() == 1
    assert (
        results_by_column["user_id"].message
        == "Column 'user_id' is not nullable but contains 1 null value(s)"
    )


def test_nullability_validator_allows_nulls_for_nullable_column(spark):
    schema = StructType(
        [
            StructField(
                "days_since_prior_order",
                IntegerType(),
                True,
            ),
        ]
    )
    df = spark.createDataFrame(
        [
            (None,),
            (7,),
            (None,),
        ],
        schema=schema,
    )

    contract = {
        "schema": [
            {
                "name": "days_since_prior_order",
                "type": "integer",
                "nullable": True,
            },
        ]
    }

    results = validate_nullability(df, contract=contract)[0]

    assert results.passed is True
    assert results.failed_count == 0
    assert results.invalid_rows is None
    assert results.message == "Column: 'days_since_prior_order' allows null values"


def test_nullability_validator_rejects_invalid_nullable_value(spark):
    df = spark.createDataFrame(
        [(1,)],
        ["order_id"],
    )

    contract = {
        "schema": [
            {
                "name": "order_id",
                "type": "integer",
                "nullable": "false",
            },
        ]
    }

    with pytest.raises(
        InvalidConstraintError,
        match="must have a boolean",
    ):
        validate_nullability(
            df,
            contract=contract,
        )


def test_nullability_validator_invalid_rows_limit(spark):
    schema = StructType(
        [
            StructField(
                "days_since_prior_order",
                IntegerType(),
                True,
            ),
        ]
    )
    df = spark.createDataFrame([(None,) for _ in range(30)], schema=schema)

    contract = {
        "schema": [
            {"name": "days_since_prior_order", "type": "integer", "nullable": False}
        ]
    }

    results = validate_nullability(df, contract=contract)[0]

    assert results.invalid_rows is not None
    assert results.invalid_rows.count() == 20
    assert results.failed_count == 30
