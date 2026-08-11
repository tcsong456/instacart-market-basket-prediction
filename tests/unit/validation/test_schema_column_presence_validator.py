import pytest

from instacart_etl_rnn.validation.exceptions import InvalidContractError
from instacart_etl_rnn.validation.schema import (
    validate_column_presence,
)


def test_column_presence_passes_when_columns_match(spark):
    df = spark.createDataFrame(
        [(1, "prior")],
        "order_id INT, eval_set STRING",
    )

    contract = {
        "dataset": {
            "allow_extra_columns": False,
        },
        "schema": [
            {"name": "order_id"},
            {"name": "eval_set"},
        ],
    }

    result = validate_column_presence(
        df,
        contract=contract,
    )

    assert result.passed
    assert result.failed_count == 0
    assert result.metadata["missing_columns"] == []
    assert result.metadata["unexpected_columns"] == []


def test_column_presence_detects_missing_columns(spark):
    df = spark.createDataFrame(
        [(1,)],
        "order_id INT",
    )

    contract = {
        "schema": [
            {"name": "order_id"},
            {"name": "eval_set"},
            {"name": "user_id"},
        ]
    }

    result = validate_column_presence(
        df,
        contract=contract,
    )

    assert result.passed is False
    assert result.failed_count == 2

    assert result.metadata["missing_columns"] == [
        "eval_set",
        "user_id",
    ]

    assert "eval_set" in result.message
    assert "user_id" in result.message


def test_column_presence_detects_unexpected_columns(spark):
    df = spark.createDataFrame(
        [(1, "prior", "extra")],
        "order_id INT, eval_set STRING, junk STRING",
    )

    contract = {
        "dataset": {
            "allow_extra_columns": False,
        },
        "schema": [
            {"name": "order_id"},
            {"name": "eval_set"},
        ],
    }

    result = validate_column_presence(
        df,
        contract=contract,
    )

    assert result.passed is False
    assert result.failed_count == 1
    assert result.metadata["unexpected_columns"] == ["junk"]


def test_column_presence_allows_extra_columns_when_configured(spark):
    df = spark.createDataFrame(
        [(1, "prior", "extra")],
        "order_id INT, eval_set STRING, junk STRING",
    )

    contract = {
        "dataset": {
            "allow_extra_columns": True,
        },
        "schema": [
            {"name": "order_id"},
            {"name": "eval_set"},
        ],
    }

    result = validate_column_presence(
        df,
        contract=contract,
    )

    assert result.passed
    assert result.failed_count == 0
    assert result.metadata["unexpected_columns"] == ["junk"]

    assert result.message == "all required columns are present"


@pytest.mark.parametrize(
    "allow_extra_columns",
    [
        "false",
        "true",
        0,
        1,
        None,
        [],
    ],
)
def test_column_presence_rejects_non_boolean_allow_extra_columns(
    spark,
    allow_extra_columns,
):
    df = spark.createDataFrame(
        [(1,)],
        "order_id INT",
    )

    contract = {
        "dataset": {
            "allow_extra_columns": allow_extra_columns,
        },
        "schema": [
            {"name": "order_id"},
        ],
    }

    with pytest.raises(
        InvalidContractError,
        match="allow_extra_columns.*boolean",
    ):
        validate_column_presence(
            df,
            contract=contract,
        )


def test_column_presence_rejects_duplicate_contract_columns(spark):
    df = spark.createDataFrame(
        [(1,)],
        "order_id INT",
    )

    contract = {
        "schema": [
            {"name": "order_id"},
            {"name": "order_id"},
        ],
    }

    with pytest.raises(
        InvalidContractError,
        match="Duplicate columns in contract schema",
    ):
        validate_column_presence(
            df,
            contract=contract,
        )


def test_column_presence_detects_duplicate_dataframe_columns(spark):
    source_df = spark.createDataFrame(
        [(1, 2)],
        "a INT, b INT",
    )

    df = source_df.selectExpr(
        "a AS value",
        "b AS value",
    )

    contract = {
        "schema": [
            {"name": "value"},
        ],
    }

    result = validate_column_presence(
        df,
        contract=contract,
    )

    assert result.passed is False
    assert result.failed_count == 1
    assert result.metadata["duplicate_columns"] == ["value"]


def test_column_presence_reports_missing_and_unexpected_columns(spark):
    df = spark.createDataFrame(
        [(1, "extra")],
        "order_id INT, junk STRING",
    )

    contract = {
        "dataset": {
            "allow_extra_columns": False,
        },
        "schema": [
            {"name": "order_id"},
            {"name": "eval_set"},
        ],
    }

    result = validate_column_presence(
        df,
        contract=contract,
    )

    assert not result.passed
    assert result.failed_count == 2

    assert result.metadata["missing_columns"] == [
        "eval_set",
    ]
    assert result.metadata["unexpected_columns"] == [
        "junk",
    ]

    assert "missing columns: eval_set" in result.message
    assert "unexpected columns: junk" in result.message


def test_column_presence_disallows_extra_columns_by_default(spark):
    df = spark.createDataFrame(
        [(1, "extra")],
        "order_id INT, junk STRING",
    )

    contract = {
        "schema": [
            {"name": "order_id"},
        ],
    }

    result = validate_column_presence(
        df,
        contract=contract,
    )

    assert not result.passed
    assert result.failed_count == 1
    assert result.metadata["unexpected_columns"] == [
        "junk",
    ]
