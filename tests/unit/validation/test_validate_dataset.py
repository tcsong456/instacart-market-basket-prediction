import logging

import pytest

from instacart_etl_rnn.validation.dataset import validate_dataset
from instacart_etl_rnn.validation.exceptions import DataValidationError
from instacart_etl_rnn.validation.models import (
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)


def test_validate_dataset_fails_fast_on_column_presence(
    mocker,
    caplog,
):
    df = mocker.sentinel.df

    contract = {
        "dataset": {
            "name": "orders",
        },
        "schema": [],
    }

    presence_result = ValidationResult(
        rule_name="column_presence",
        passed=False,
        failed_count=1,
        message="Missing column: user_id",
    )

    mocked_presence = mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_column_presence",
        return_value=presence_result,
    )

    mocked_datatype = mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_column_datatype",
    )

    mocked_metrics = mocker.patch(
        "instacart_etl_rnn.validation.dataset._run_column_metrics",
    )

    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(DataValidationError) as exc_info:
            validate_dataset(
                df,
                contract=contract,
            )

    mocked_presence.assert_called_once_with(
        df,
        contract=contract,
    )

    mocked_datatype.assert_not_called()
    mocked_metrics.assert_not_called()

    assert exc_info.value.report.dataset_name == "orders"
    assert exc_info.value.report.results == []

    assert "Missing column: user_id" in caplog.text


def test_validate_dataset_fails_fast_on_column_datatype(
    mocker,
    caplog,
):
    df = mocker.sentinel.df

    contract = {
        "dataset": {
            "name": "orders",
        },
        "schema": [],
    }

    presence_result = ValidationResult(
        rule_name="column_presence",
        passed=True,
        failed_count=0,
        message="Column presence passed",
    )

    datatype_result = ValidationResult(
        rule_name="column_datatype",
        passed=False,
        failed_count=1,
        message="order_id has invalid datatype",
    )

    mocked_presence = mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_column_presence",
        return_value=presence_result,
    )

    mocked_datatype = mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_column_datatype",
        return_value=datatype_result,
    )

    mocked_metrics = mocker.patch(
        "instacart_etl_rnn.validation.dataset._run_column_metrics",
    )

    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(DataValidationError) as exc_info:
            validate_dataset(
                df,
                contract=contract,
            )

    mocked_presence.assert_called_once_with(
        df,
        contract=contract,
    )

    mocked_datatype.assert_called_once_with(
        df,
        contract=contract,
    )

    mocked_metrics.assert_not_called()

    assert exc_info.value.report.dataset_name == "orders"
    assert exc_info.value.report.results == []

    assert "order_id has invalid datatype" in caplog.text


def test_validate_dataset_raises_with_validation_results_on_hard_failure(
    mocker,
):
    raw_uniqueness_result = ValidationResult(
        rule_name="order_id.uniqueness",
        failed_count=2,
    )

    failed_uniqueness_result = ValidationResult(
        rule_name="order_id.uniqueness",
        passed=False,
        status=ValidationStatus.FAILED,
        severity=ValidationSeverity.CRITICAL,
        failed_count=2,
    )

    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_column_presence",
        return_value=ValidationResult(
            passed=True,
            status=ValidationStatus.PASSED,
        ),
    )

    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_column_datatype",
        return_value=ValidationResult(
            passed=True,
            status=ValidationStatus.PASSED,
        ),
    )

    mocker.patch(
        "instacart_etl_rnn.validation.dataset._run_column_metrics",
        return_value=([], 100),
    )

    mocker.patch(
        "instacart_etl_rnn.validation.dataset._run_uniqueness_validators",
        return_value=[
            raw_uniqueness_result,
        ],
    )

    mocker.patch(
        "instacart_etl_rnn.validation.dataset._apply_strict_result_status",
        return_value=failed_uniqueness_result,
    )

    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_foreign_keys",
        return_value=[],
    )

    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_row_logic",
        return_value=[],
    )

    mocker.patch(
        "instacart_etl_rnn.validation.dataset._log_validation_results",
    )

    with pytest.raises(DataValidationError) as exc_info:
        validate_dataset(
            mocker.sentinel.df,
            contract={
                "dataset": {
                    "name": "orders",
                },
                "schema": [],
            },
        )

    report = exc_info.value.report

    assert report.dataset_name == "orders"
    assert report.results == [
        failed_uniqueness_result,
    ]
