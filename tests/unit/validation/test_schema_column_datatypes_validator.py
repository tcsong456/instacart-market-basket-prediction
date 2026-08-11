from pyspark.sql.types import StringType

from instacart_etl_rnn.validation.schema import validate_column_datatype


def test_column_datatype_passes_for_compatible_schema(spark):
    df = spark.createDataFrame(
        [(1, 10, "prior")],
        "order_id INT, user_id BIGINT, eval_set STRING",
    )

    contract = {
        "schema": [
            {
                "name": "order_id",
                "type": "integer",
            },
            {
                "name": "user_id",
                "type": "integer",
            },
            {
                "name": "eval_set",
                "type": "string",
            },
        ]
    }

    result = validate_column_datatype(
        df,
        contract=contract,
    )

    assert result.passed
    assert result.failed_count == 0
    assert result.metadata["mismatches"] == []


def test_column_datatype_reports_mismatched_columns(spark):
    df = spark.createDataFrame(
        [(1, "10", 5)],
        "order_id INT, user_id STRING, eval_set INT",
    )

    contract = {
        "schema": [
            {
                "name": "order_id",
                "type": "integer",
            },
            {
                "name": "user_id",
                "type": "integer",
            },
            {
                "name": "eval_set",
                "type": "string",
            },
        ]
    }

    result = validate_column_datatype(
        df,
        contract=contract,
    )

    assert not result.passed
    assert result.failed_count == 2

    mismatches = result.metadata["mismatches"]

    assert {mismatch["column_name"] for mismatch in mismatches} == {
        "user_id",
        "eval_set",
    }

    mismatch_by_column = {mismatch["column_name"]: mismatch for mismatch in mismatches}

    assert isinstance(
        mismatch_by_column["user_id"]["actual_datatype"],
        StringType,
    )

    assert mismatch_by_column["user_id"]["expected_datatype"] == "integer"


def test_column_datatype_does_not_report_missing_columns(spark):
    df = spark.createDataFrame(
        [(1,)],
        "order_id INT",
    )

    contract = {
        "schema": [
            {
                "name": "order_id",
                "type": "integer",
            },
            {
                "name": "eval_set",
                "type": "string",
            },
        ]
    }

    result = validate_column_datatype(
        df,
        contract=contract,
    )

    assert result.passed
    assert result.failed_count == 0
    assert result.metadata["mismatches"] == []


def test_column_datatype_supports_array_columns(spark):
    df = spark.createDataFrame(
        [
            (1, [1, 2, 3]),
        ],
        "order_id INT, product_ids ARRAY<INT>",
    )

    contract = {
        "schema": [
            {
                "name": "order_id",
                "type": "integer",
            },
            {
                "name": "product_ids",
                "type": "array<integer>",
            },
        ]
    }

    result = validate_column_datatype(
        df,
        contract=contract,
    )

    assert result.passed
    assert result.failed_count == 0
