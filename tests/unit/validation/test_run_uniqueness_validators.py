from unittest.mock import call

import pytest

from instacart_etl_rnn.validation.dataset import _run_uniqueness_validators
from instacart_etl_rnn.validation.exceptions import InvalidContractError


def test_run_uniqueness_validators_runs_unique_column(spark, mocker):
    df = spark.createDataFrame([(1,), (2,)], ["order_id"])

    expected_result = object()
    mocked_validate = mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_uniqueness",
        return_value=expected_result,
    )

    contract = {
        "schema": [
            {
                "name": "order_id",
                "constraints": {
                    "unique": True,
                },
            },
            {
                "name": "user_id",
                "constraints": {
                    "unique": False,
                },
            },
        ]
    }

    results = _run_uniqueness_validators(df, contract=contract)

    assert results == [expected_result]
    mocked_validate.assert_called_once_with(df, columns=["order_id"])


def test_run_uniqueness_validators_runs_each_unique_column(spark, mocker):
    df = spark.createDataFrame([(1, 10), (2, 20)], ["order_id", "user_id"])

    first_result = object()
    second_result = object()
    mocked_validate = mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_uniqueness",
        side_effect=[first_result, second_result],
    )

    contract = {
        "schema": [
            {
                "name": "order_id",
                "constraints": {"unique": True},
            },
            {"name": "user_id", "constraints": {"unique": True}},
        ]
    }

    results = _run_uniqueness_validators(df, contract=contract)
    assert results == [first_result, second_result]
    assert mocked_validate.call_args_list == [
        call(df, columns=["order_id"]),
        call(df, columns=["user_id"]),
    ]


def test_run_uniqueness_validators_does_not_repeat_same_single_column_grain(
    spark, mocker
):
    df = spark.createDataFrame([(1,), (2,)], ["order_id"])

    expected_result = object()
    mocked_validate = mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_uniqueness",
        return_value=expected_result,
    )

    contract = {
        "dataset": {"grain": ["order_id"]},
        "schema": [{"name": "order_id", "constraints": {"unique": True}}],
    }

    results = _run_uniqueness_validators(df, contract=contract)

    assert results == [expected_result]
    mocked_validate.assert_called_once_with(df, columns=["order_id"])


def test_run_uniqueness_validators_does_not_run_empty_unique(spark, mocker):
    df = spark.createDataFrame([(1,)], ["order_id"])

    mocked_validate = mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_uniqueness",
    )

    contract = {"schema": [{"name": "order_id", "constraints": {}}]}

    results = _run_uniqueness_validators(df, contract=contract)

    mocked_validate.assert_not_called()
    assert results == []


def test_run_uniqueness_validators_runs_column_and_grain(spark, mocker):
    df = spark.createDataFrame([(1, 10), (2, 20)], ["order_id", "user_id"])

    column_result = object()
    grain_result = object()
    mocked_validate = mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_uniqueness",
        side_effect=[column_result, grain_result],
    )

    contract = {
        "dataset": {"grain": ["order_id", "user_id"]},
        "schema": [{"name": "order_id", "constraints": {"unique": True}}],
    }

    results = _run_uniqueness_validators(df, contract=contract)

    assert results == [column_result, grain_result]
    assert mocked_validate.call_args_list == [
        call(df, columns=["order_id"]),
        call(df, columns=["order_id", "user_id"]),
    ]


def test_run_uniqueness_validators_runs_grain_only(spark, mocker):
    df = spark.createDataFrame([(1, 1), (1, 2)], ["order_id", "order_number"])

    grain_result = object()
    mocked_validate = mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_uniqueness",
        return_value=grain_result,
    )

    contract = {
        "dataset": {"grain": ["order_id", "order_number"]},
        "schema": [{"name": "order_id", "constraints": {}}],
    }

    results = _run_uniqueness_validators(df, contract=contract)

    assert results == [grain_result]
    mocked_validate.assert_called_once_with(df, columns=["order_id", "order_number"])


@pytest.mark.parametrize(
    "unique",
    [
        "true",
        1,
        [],
        {},
        None,
    ],
)
def test_run_uniqueness_validators_rejects_non_boolean_unique(
    spark,
    unique,
):
    df = spark.createDataFrame(
        [(1,)],
        ["order_id"],
    )

    contract = {
        "schema": [
            {
                "name": "order_id",
                "constraints": {
                    "unique": unique,
                },
            },
        ]
    }

    with pytest.raises(
        InvalidContractError,
        match="'unique'.*must be a boolean",
    ):
        _run_uniqueness_validators(
            df,
            contract=contract,
        )


@pytest.mark.parametrize(
    "grain",
    [
        "order_id",
        [],
        [1],
        ["order_id", 2],
        {},
    ],
)
def test_run_uniqueness_validators_rejects_invalid_grain(
    spark,
    grain,
):
    df = spark.createDataFrame(
        [(1,)],
        ["order_id"],
    )

    contract = {
        "dataset": {
            "grain": grain,
        },
        "schema": [
            {
                "name": "order_id",
                "constraints": {},
            },
        ],
    }

    with pytest.raises(
        InvalidContractError,
        match="The dataset grain must be a non-empty list of column names",
    ):
        _run_uniqueness_validators(
            df,
            contract=contract,
        )
