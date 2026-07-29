import pytest

from instacart_etl_rnn.validation.exceptions import InvalidContractError
from instacart_etl_rnn.validation.row_logic import validate_row_logic


def test_validate_row_logic_all_rules_pass(
    spark,
    orders_schema,
    row_logic_contract,
):
    df = spark.createDataFrame(
        [
            (1, 1, "prior", None),
            (1, 2, "prior", 7.0),
            (1, 3, "train", 4.0),
            (2, 1, "prior", None),
            (2, 2, "test", 5.0),
        ],
        schema=orders_schema,
    )

    results = validate_row_logic(
        df,
        row_logic_contract,
    )

    assert len(results) == 5

    for result in results:
        assert result.category == "row_logic"
        assert result.passed is True
        assert result.failed_count == 0
        assert result.invalid_rows is None
        assert (
            result.message
            == f"Rule: {result.rule_name} complies with the data contract rule"
        )


def test_validate_row_logic_returns_expected_failure_counts(
    spark,
    orders_schema,
    row_logic_contract,
):
    df = spark.createDataFrame(
        [
            (1, 1, "prior", 5.0),
            (1, 2, "prior", None),
            (1, 3, "prior", 4.0),
            (2, 1, "prior", None),
            (2, 3, "train", 6.0),
        ],
        schema=orders_schema,
    )

    results = validate_row_logic(
        df,
        row_logic_contract,
    )

    results_by_name = {result.rule_name: result for result in results}

    expected_failed_counts = {
        "first_order_has_no_prior_interval": 1,
        "later_orders_must_have_prior_interval": 1,
        "last_order_per_user_is_train_or_test": 1,
        "only_one_train_or_test_per_user": 3,
        "contiguous_order_numbers": 2,
    }

    assert set(results_by_name) == set(expected_failed_counts)

    for rule_name, expected_count in expected_failed_counts.items():
        result = results_by_name[rule_name]

        assert result.passed is False
        assert result.failed_count == expected_count
        assert result.invalid_rows is not None
        assert result.message == f"Rule: {rule_name} failed the data contract rule"


def test_validate_row_logic_returns_correct_invalid_rows(
    spark,
    orders_schema,
    row_logic_contract,
):
    df = spark.createDataFrame(
        [
            (1, 1, "prior", None),
            (1, 3, "train", 5),
            (2, 1, "prior", 10),
            (2, 2, "test", 20),
        ],
        schema=orders_schema,
    )

    results = validate_row_logic(
        df,
        row_logic_contract,
    )

    results_by_name = {result.rule_name: result for result in results}

    result_1 = results_by_name["first_order_has_no_prior_interval"]
    assert result_1.invalid_rows is not None
    invalid_rows_1 = result_1.invalid_rows.collect()
    assert len(invalid_rows_1) == 1
    row = invalid_rows_1[0]
    assert row["user_id"] == 2
    assert row["order_number"] == 1
    assert row["days_since_prior_order"] == 10

    result_2 = results_by_name["contiguous_order_numbers"]
    assert result_2.invalid_rows is not None

    actual_rows = {
        (
            row["user_id"],
            row["order_number"],
            row["days_since_prior_order"],
        )
        for row in result_2.invalid_rows.collect()
    }

    expected_rows = {
        (1, 1, None),
        (1, 3, 5.0),
    }

    assert actual_rows == expected_rows

    passed_rules = [
        "later_orders_must_have_prior_interval",
        "last_order_per_user_is_train_or_test",
        "only_one_train_or_test_per_user",
    ]
    for rule in passed_rules:
        result = results_by_name[rule]
        assert result.passed


def test_validate_row_logic_limits_invalid_rows(
    spark,
    orders_schema,
    row_logic_contract,
):
    df = spark.createDataFrame(
        [(user_id, 1, "prior", 10.0) for user_id in range(1, 31)], schema=orders_schema
    )

    results = validate_row_logic(
        df,
        row_logic_contract,
    )

    results_by_name = {result.rule_name: result for result in results}

    result = results_by_name["first_order_has_no_prior_interval"]

    assert result.failed_count == 30
    assert result.invalid_rows.count() == 20


def test_validate_row_logic_empty_dataframe_passes(
    spark,
    orders_schema,
    row_logic_contract,
):
    df = spark.createDataFrame(
        [],
        schema=orders_schema,
    )

    results = validate_row_logic(
        df,
        row_logic_contract,
    )

    assert len(results) == 5

    for result in results:
        assert result.passed is True
        assert result.failed_count == 0
        assert result.invalid_rows is None


def test_validate_row_logic_empty_rule(spark, orders_schema):
    df = spark.createDataFrame([(1, 1, "prior", None)], schema=orders_schema)

    contract = {"rules": []}

    results = validate_row_logic(
        df,
        contract,
    )

    assert results == []


@pytest.mark.parametrize(
    ("contract", "exception_message"),
    [
        (
            {
                "derived_fields": [
                    {
                        "name": "train_or_test_count",
                        "type": "integer",
                        "column": "order_number",
                    }
                ]
            },
            "Missing both aggregation and expression.",
        ),
        (
            {
                "derived_fields": [
                    {
                        "name": "train_or_test_count",
                        "type": "integer",
                        "aggregation": "min_max",
                    }
                ]
            },
            "Aggregation type",
        ),
        (
            {
                "derived_fields": [
                    {
                        "name": "train_or_test_count",
                        "type": "integer",
                        "aggregation": "min",
                    }
                ]
            },
            "does not provide its aggregated column",
        ),
        (
            {
                "derived_fields": [
                    {
                        "name": "train_or_test_count",
                        "column": "order_number",
                        "aggregation": "min",
                        "expression": "order_number = user_max_order_number",
                    }
                ]
            },
            "Only one of aggregation or expression should be given",
        ),
        (
            {
                "rules": [
                    {
                        "expression": ("order_number <> 1 OR "
                                       "days_since_prior_order is NULL")
                    }
                ]
            },
            "Every rule must have a name",
        ),
        (
            {"rules": [{"name": "last_order_per_user_is_train_or_test"}]},
            "does not have its expression",
        ),
    ],
)
def test_validate_row_logic_invalid_contract(
    spark, contract, exception_message, orders_schema
):
    df = spark.createDataFrame([(1, 1, "prior", None)], schema=orders_schema)

    with pytest.raises(InvalidContractError, match=exception_message):
        validate_row_logic(df, contract)
