import pytest

from instacart_etl_rnn.validation.dataset import validate_dataset
from instacart_etl_rnn.validation.exceptions import DataValidationError


def test_user_data_complies_with_contract(spark, user_data_contract, user_data_schema):
    df = spark.createDataFrame(
        [
            (
                1,
                "train",
                [1, 2, 3],
                [1, 3, 5],
                [10, 14, 18],
                [None, 7.0, 4.0],
                ["0_0_1", "1_0", "1_1_0"],
            ),
            (
                2,
                "test",
                [1, 2],
                [0, 6],
                [9, 21],
                [None, 12.0],
                ["0_1", "1_1"],
            ),
            (
                3,
                "train",
                [1, 2, 3, 4],
                [2, 2, 4, 5],
                [8, 12, 16, 20],
                [None, 3.0, 10.0, 30.0],
                ["0", "1_0", "1_1", "0_1"],
            ),
        ],
        schema=user_data_schema,
    )

    report = validate_dataset(
        df,
        contract=user_data_contract,
    )

    assert report.can_continue
    assert not report.has_errors

    for result in report.results:
        assert result.failed_count == 0


def test_user_data_fails_the_contract(spark, user_data_contract, user_data_schema):
    invalid_user_data_df = spark.createDataFrame(
        [
            (
                1,
                "validation",
                [1, 3],
                [1, 8],
                [10, 25],
                [5.0, 35.0],
                ["0_1", "1_2"],
            ),
        ],
        schema=user_data_schema,
    )

    with pytest.raises(DataValidationError) as exc_info:
        validate_dataset(
            invalid_user_data_df,
            contract=user_data_contract,
        )

    report = exc_info.value.report

    failed_rules = {
        result.rule_name for result in report.results if result.failed_count > 0
    }

    assert "eval_set.allowed_values" in failed_rules
    assert "order_numbers_are_contiguous" in failed_rules
    assert "order_dows_are_valid" in failed_rules
    assert "order_hours_are_valid" in failed_rules
    assert "first_order_has_no_prior_interval" in failed_rules
    assert "reorder_values_are_binary" in failed_rules
