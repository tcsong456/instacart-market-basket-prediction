import logging
from unittest.mock import call

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


def test_validate_dataset_returns_report_when_all_validations_pass(
    mocker,
):
    df = mocker.sentinel.df
    users_df = mocker.sentinel.users_df

    contract = {
        "dataset": {
            "name": "orders",
        },
        "schema": [],
    }

    presence_result = ValidationResult(
        passed=True,
        status=ValidationStatus.PASSED,
    )

    datatype_result = ValidationResult(
        passed=True,
        status=ValidationStatus.PASSED,
    )

    column_result = ValidationResult(
        rule_name="order_id.range",
        passed=True,
        status=ValidationStatus.PASSED,
        failed_count=0,
    )

    uniqueness_raw = ValidationResult(
        rule_name="order_id.uniqueness",
        failed_count=0,
    )

    ri_raw = ValidationResult(
        rule_name="orders.user_id.foreign_key",
        failed_count=0,
    )

    row_logic_raw = ValidationResult(
        rule_name="first_order_rule",
        failed_count=0,
    )

    uniqueness_result = ValidationResult(
        rule_name="order_id.uniqueness",
        passed=True,
        status=ValidationStatus.PASSED,
        severity=ValidationSeverity.INFO,
        failed_count=0,
    )

    ri_result = ValidationResult(
        rule_name="orders.user_id.foreign_key",
        passed=True,
        status=ValidationStatus.PASSED,
        severity=ValidationSeverity.INFO,
        failed_count=0,
    )

    row_logic_result = ValidationResult(
        rule_name="first_order_rule",
        passed=True,
        status=ValidationStatus.PASSED,
        severity=ValidationSeverity.INFO,
        failed_count=0,
    )

    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_column_presence",
        return_value=presence_result,
    )

    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_column_datatype",
        return_value=datatype_result,
    )

    mocked_metrics = mocker.patch(
        "instacart_etl_rnn.validation.dataset._run_column_metrics",
        return_value=(
            [column_result],
            100,
        ),
    )

    mocked_uniqueness = mocker.patch(
        "instacart_etl_rnn.validation.dataset._run_uniqueness_validators",
        return_value=[uniqueness_raw],
    )

    mocked_fk = mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_foreign_keys",
        return_value=[ri_raw],
    )

    mocked_row_logic = mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_row_logic",
        return_value=[row_logic_raw],
    )

    mocked_apply = mocker.patch(
        "instacart_etl_rnn.validation.dataset._apply_strict_result_status",
        side_effect=[
            uniqueness_result,
            ri_result,
            row_logic_result,
        ],
    )

    mocked_log = mocker.patch(
        "instacart_etl_rnn.validation.dataset._log_validation_results",
    )

    report = validate_dataset(
        df,
        contract=contract,
        reference_datasets={
            "users": users_df,
        },
    )

    assert report.dataset_name == "orders"

    assert report.results == [
        column_result,
        uniqueness_result,
        ri_result,
        row_logic_result,
    ]

    mocked_metrics.assert_called_once_with(
        df,
        contract=contract,
    )

    mocked_uniqueness.assert_called_once_with(
        df,
        contract=contract,
    )

    mocked_fk.assert_called_once_with(
        child_df=df,
        contract=contract,
        datasets={
            "users": users_df,
        },
    )

    mocked_row_logic.assert_called_once_with(
        df,
        contract=contract,
    )

    assert mocked_apply.call_args_list == [
        call(
            uniqueness_raw,
            total_rows=100,
            severity=ValidationSeverity.CRITICAL,
        ),
        call(
            ri_raw,
            total_rows=100,
            severity=ValidationSeverity.CRITICAL,
        ),
        call(
            row_logic_raw,
            total_rows=100,
            severity=ValidationSeverity.CRITICAL,
        ),
    ]

    mocked_log.assert_called_once_with(
        report.results,
    )


def test_validate_dataset_uses_empty_reference_datasets_by_default(
    mocker,
):
    df = mocker.sentinel.df

    contract = {
        "dataset": {
            "name": "orders",
        },
        "schema": [],
    }

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
        return_value=([], 10),
    )

    mocker.patch(
        "instacart_etl_rnn.validation.dataset._run_uniqueness_validators",
        return_value=[],
    )

    mocked_fk = mocker.patch(
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

    validate_dataset(
        df,
        contract=contract,
    )

    mocked_fk.assert_called_once_with(
        child_df=df,
        contract=contract,
        datasets={},
    )


def test_validate_dataset_returns_report_when_only_warning_exists(
    mocker,
):
    warning_result = ValidationResult(
        rule_name="order_id.range",
        passed=True,
        status=ValidationStatus.WARNING,
        severity=ValidationSeverity.WARNING,
        failed_count=1,
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
        return_value=(
            [warning_result],
            100,
        ),
    )

    mocker.patch(
        "instacart_etl_rnn.validation.dataset._run_uniqueness_validators",
        return_value=[],
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

    report = validate_dataset(
        mocker.sentinel.df,
        contract={
            "dataset": {
                "name": "orders",
            },
            "schema": [],
        },
    )

    assert report.results == [
        warning_result,
    ]

    assert report.has_errors is False


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
