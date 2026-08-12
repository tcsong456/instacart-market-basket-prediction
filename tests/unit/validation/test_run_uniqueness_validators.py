from unittest.mock import call

import pytest

from instacart_etl_rnn.validation.dataset import (
    _run_uniqueness_validators,
)
from instacart_etl_rnn.validation.exceptions import (
    InvalidContractError,
)


def test_run_uniqueness_validators_runs_unique_columns_and_grain(
    mocker,
):
    df = mocker.sentinel.df

    order_id_result = mocker.sentinel.order_id_result
    grain_result = mocker.sentinel.grain_result

    mocked_validate = mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_uniqueness",
        side_effect=[
            order_id_result,
            grain_result,
        ],
    )

    contract = {
        "dataset": {
            "grain": [
                "order_id",
                "product_id",
            ],
        },
        "schema": [
            {
                "name": "order_id",
                "constraints": {
                    "unique": True,
                },
            },
            {
                "name": "product_id",
                "constraints": {
                    "unique": False,
                },
            },
        ],
    }

    results = _run_uniqueness_validators(
        df,
        contract=contract,
    )

    assert results == [
        order_id_result,
        grain_result,
    ]

    assert mocked_validate.call_args_list == [
        call(
            df,
            columns=["order_id"],
        ),
        call(
            df,
            columns=[
                "order_id",
                "product_id",
            ],
        ),
    ]


def test_run_uniqueness_validators_does_not_repeat_single_column_grain(
    mocker,
):
    df = mocker.sentinel.df
    expected_result = mocker.sentinel.result

    mocked_validate = mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_uniqueness",
        return_value=expected_result,
    )

    contract = {
        "dataset": {
            "grain": ["order_id"],
        },
        "schema": [
            {
                "name": "order_id",
                "constraints": {
                    "unique": True,
                },
            },
        ],
    }

    results = _run_uniqueness_validators(
        df,
        contract=contract,
    )

    assert results == [
        expected_result,
    ]

    mocked_validate.assert_called_once_with(
        df,
        columns=["order_id"],
    )


def test_run_uniqueness_validators_skips_non_unique_columns(
    mocker,
):
    df = mocker.sentinel.df

    mocked_validate = mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_uniqueness",
    )

    contract = {
        "schema": [
            {
                "name": "order_id",
                "constraints": {
                    "unique": False,
                },
            },
            {
                "name": "user_id",
            },
        ],
    }

    results = _run_uniqueness_validators(
        df,
        contract=contract,
    )

    assert results == []
    mocked_validate.assert_not_called()


@pytest.mark.parametrize(
    "invalid_unique",
    [
        1,
        0,
        "true",
        "false",
        [],
        {},
        None,
    ],
)
def test_run_uniqueness_validators_rejects_non_boolean_unique(
    mocker,
    invalid_unique,
):
    contract = {
        "schema": [
            {
                "name": "order_id",
                "constraints": {
                    "unique": invalid_unique,
                },
            },
        ],
    }

    with pytest.raises(
        InvalidContractError,
        match="'unique' for column 'order_id' must be a boolean",
    ):
        _run_uniqueness_validators(
            mocker.sentinel.df,
            contract=contract,
        )


@pytest.mark.parametrize(
    "invalid_grain",
    [
        [],
        "order_id",
        [""],
        ["order_id", ""],
        [1],
    ],
)
def test_run_uniqueness_validators_rejects_invalid_grain(
    mocker,
    invalid_grain,
):
    contract = {
        "dataset": {
            "grain": invalid_grain,
        },
        "schema": [],
    }

    with pytest.raises(
        InvalidContractError,
        match="The dataset grain must be a non-empty list of column names",
    ):
        _run_uniqueness_validators(
            mocker.sentinel.df,
            contract=contract,
        )


def test_run_uniqueness_validators_allows_missing_grain(
    mocker,
):
    mocked_validate = mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_uniqueness",
    )

    results = _run_uniqueness_validators(
        mocker.sentinel.df,
        contract={
            "dataset": {},
            "schema": [],
        },
    )

    assert results == []
    mocked_validate.assert_not_called()
