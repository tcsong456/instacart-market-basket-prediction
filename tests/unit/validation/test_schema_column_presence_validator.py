import pytest
from pyspark.sql import Row

from instacart_etl.validation.schema import validate_columns


def build_contract(allow_extra_columns: bool = False):
    return {
        "dataset": {"allow_extra_columns": allow_extra_columns},
        "schema": [{"name": "order_id"}, {"name": "user_id"}, {"name": "eval_set"}],
    }


@pytest.mark.parametrize(
    (
        "raw_data",
        "allow_extra_columns",
        "expected_passed",
        "expected_failed_count",
        "expected_unexpected_columns",
        "expected_missing_columns",
        "expected_actual_columns",
        "expected_message",
    ),
    [
        (
            {"order_id": 10, "user_id": 5, "eval_set": "prior"},
            False,
            True,
            0,
            [],
            [],
            ["order_id", "user_id", "eval_set"],
            "all required columns are present",
        ),
        (
            {"order_id": 10, "user_id": 5},
            False,
            False,
            1,
            [],
            ["eval_set"],
            ["order_id", "user_id"],
            "missing columns: eval_set; unexpected columns: none",
        ),
        (
            {"order_id": 10, "user_id": 5, "eval_set": "prior", "order_number": 100},
            False,
            False,
            1,
            ["order_number"],
            [],
            ["order_id", "user_id", "eval_set", "order_number"],
            "missing columns: none; unexpected columns: order_number",
        ),
        (
            {"order_id": 10, "user_id": 5, "eval_set": "prior", "order_number": 100},
            True,
            True,
            0,
            ["order_number"],
            [],
            ["order_id", "user_id", "eval_set", "order_number"],
            "all required columns are present",
        ),
        (
            {"order_id": 10, "user_id": 5, "order_number": 100},
            False,
            False,
            2,
            ["order_number"],
            ["eval_set"],
            ["order_id", "user_id", "order_number"],
            "missing columns: eval_set; unexpected columns: order_number",
        ),
        (
            {"order_id": 10, "user_id": 5, "order_number": 100},
            True,
            False,
            1,
            ["order_number"],
            ["eval_set"],
            ["order_id", "user_id", "order_number"],
            "missing columns: eval_set; unexpected columns: order_number (ignored)",
        ),
    ],
)
def test_validate_columns(
    spark,
    raw_data,
    allow_extra_columns,
    expected_passed,
    expected_failed_count,
    expected_unexpected_columns,
    expected_missing_columns,
    expected_actual_columns,
    expected_message,
):
    df = spark.createDataFrame([Row(**raw_data)])
    contract = build_contract(allow_extra_columns)

    result = validate_columns(df, contract=contract)

    assert result.passed is expected_passed
    assert result.failed_count == expected_failed_count
    assert result.metadata["unexpected_columns"] == sorted(expected_unexpected_columns)
    assert result.metadata["missing_columns"] == sorted(expected_missing_columns)
    assert result.message == expected_message
    assert result.invalid_rows is None
    assert result.rule_name == "column_presence"
    assert result.category == "schema"
    assert result.metadata["actual_columns"] == sorted(expected_actual_columns)
    assert result.metadata["expected_columns"] == sorted(
        [s["name"] for s in contract["schema"]]
    )


def build_contract_without_dataset_as_key():
    return {"schema": [{"name": "order_id"}, {"name": "user_id"}, {"name": "eval_set"}]}


def test_validate_columns_with_contract_missing_dataset(spark):
    df = spark.createDataFrame(
        [Row(order_id=10, user_id=5, eval_set="prior", order_number=100)]
    )
    contract = build_contract_without_dataset_as_key()

    result = validate_columns(df, contract=contract)

    assert result.passed is False
    assert result.failed_count == 1
    assert result.metadata["unexpected_columns"] == ["order_number"]
    assert result.metadata["missing_columns"] == []
