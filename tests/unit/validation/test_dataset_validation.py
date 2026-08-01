import logging

import pytest

from instacart_etl_rnn.validation.dataset import validate_dataset
from instacart_etl_rnn.validation.exceptions import (
    DataValidationError,
    EmptyDatasetError,
)
from instacart_etl_rnn.validation.models import (
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)


def test_validate_dataset_rejects_empty_dataframe(spark):
    df = spark.createDataFrame(
        [],
        "order_id INT",
    )

    contract = {
        "dataset": {
            "name": "orders",
        },
        "schema": [
            {
                "name": "order_id",
                "type": "integer",
            },
        ],
    }

    with pytest.raises(
        EmptyDatasetError,
        match="must not be empty",
    ):
        validate_dataset(
            df,
            contract=contract,
        )


def test_validate_dataset_fails_fast_on_column_presence(spark, mocker, caplog):
    df = spark.createDataFrame([(1,)], ["order_id"])

    presence_result = ValidationResult(
        rule_name="column_presence",
        passed=False,
        message="Missing column: user_id",
    )

    mocked_presence = mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_column_presence",
        return_value=presence_result,
    )
    mocked_datatype = mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_column_datatype"
    )

    contract = {
        "dataset": {"name": "orders"},
        "schema": [{"name": "order_id"}, {"name": "user_id"}],
    }

    with caplog.at_level("CRITICAL"):
        with pytest.raises(DataValidationError) as exc_info:
            validate_dataset(df, contract=contract)

    mocked_presence.assert_called_once_with(df, contract=contract)
    mocked_datatype.assert_not_called()

    assert exc_info.value.report.dataset_name == "orders"
    assert exc_info.value.report.results == [presence_result]
    assert "Missing column: user_id" in caplog.text


def test_validate_dataset_fails_fast_on_datatype(spark, mocker):
    df = spark.createDataFrame([(1,)], ["order_id"])

    presence_result = ValidationResult(rule_name="column_presence", passed=True)
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_column_presence",
        return_value=presence_result,
    )

    datatype_result = ValidationResult(rule_name="column_types", passed=False)
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_column_datatype",
        return_value=datatype_result,
    )

    mocked_nullability = mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_nullability"
    )

    contract = {
        "dataset": {"name": "orders"},
        "schema": [{"name": "order_id"}],
    }

    with pytest.raises(DataValidationError):
        validate_dataset(df, contract=contract)

    mocked_nullability.assert_not_called()


def test_validate_dataset_returns_report_when_all_validations_pass(
    spark,
    mocker,
    caplog,
):
    df = spark.createDataFrame(
        [(1, 10)],
        ["order_id", "user_id"],
    )

    presence = ValidationResult(rule_name="column_presence", passed=True)
    datatype = ValidationResult(rule_name="column_datatype", passed=True)

    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_column_presence",
        return_value=presence,
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_column_datatype",
        return_value=datatype,
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_nullability",
        return_value=[],
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_allowed_values",
        return_value=[],
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_range",
        return_value=[],
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_array_lengths",
        return_value=[],
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

    contract = {
        "dataset": {"name": "orders"},
        "schema": [
            {"name": "order_id"},
            {"name": "user_id"},
        ],
    }

    with caplog.at_level("INFO"):
        report = validate_dataset(
            df,
            contract=contract,
        )

    assert report.dataset_name == "orders"
    assert report.results == [presence, datatype]
    assert "has passed the data validation" in caplog.text


def test_validate_dataset_logs_warning_and_returns_report(
    spark,
    mocker,
    caplog,
):
    df = spark.createDataFrame(
        [(1,), (2,)],
        ["order_id"],
    )

    presence = ValidationResult(rule_name="column_presence", passed=True)
    datatype = ValidationResult(rule_name="column_datatype", passed=True)

    warning_result = ValidationResult(
        rule_name="order_id.range",
        failed_count=1,
        passed=False,
        status=ValidationStatus.WARNING,
        severity=ValidationSeverity.WARNING,
        message="order_id.range within allowable threshold",
    )

    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_column_presence",
        return_value=presence,
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_column_datatype",
        return_value=datatype,
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_nullability",
        return_value=[],
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_allowed_values",
        return_value=[],
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_range",
        return_value=[warning_result],
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset._apply_thresholds",
        side_effect=lambda results, **kwargs: results,
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_array_lengths",
        return_value=[],
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

    contract = {
        "dataset": {"name": "orders"},
        "schema": [{"name": "order_id"}],
    }

    with caplog.at_level("WARNING"):
        report = validate_dataset(
            df,
            contract=contract,
        )

    assert warning_result in report.results
    assert "within allowable threshold" in caplog.text

    caplog.clear()

    with caplog.at_level("INFO"):
        validate_dataset(
            df,
            contract=contract,
        )

    assert any(
        record.levelno == logging.INFO
        and record.message.startswith("All column and contract rules passed")
        for record in caplog.records
    )


def test_validate_dataset_logs_error_and_raises(
    spark,
    mocker,
    caplog,
):
    df = spark.createDataFrame(
        [
            (1,),
            (2,),
        ],
        ["order_id"],
    )

    presence = ValidationResult(rule_name="column_presence", passed=True)
    datatype = ValidationResult(rule_name="column_datatype", passed=True)

    error_result = ValidationResult(
        rule_name="order_id.range",
        failed_count=1,
        passed=False,
        status=ValidationStatus.FAILED,
        severity=ValidationSeverity.ERROR,
        message="order_id.range failed",
    )

    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_column_presence",
        return_value=presence,
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_column_datatype",
        return_value=datatype,
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_nullability",
        return_value=[],
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_allowed_values",
        return_value=[],
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_range",
        return_value=[error_result],
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset._apply_thresholds",
        side_effect=lambda results, **kwargs: results,
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_array_lengths",
        return_value=[],
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

    contract = {
        "dataset": {"name": "orders"},
        "schema": [{"name": "order_id"}],
    }

    with caplog.at_level("ERROR"):
        with pytest.raises(DataValidationError) as exc_info:
            validate_dataset(
                df,
                contract=contract,
            )

    assert error_result in exc_info.value.report.results
    assert "order_id.range failed" in caplog.text


def test_validate_dataset_logs_critical_result(
    spark,
    mocker,
    caplog,
):
    df = spark.createDataFrame(
        [(1,), (2,)],
        ["order_id"],
    )

    critical_result = ValidationResult(
        rule_name="grain_uniqueness",
        failed_count=1,
        status=ValidationStatus.FAILED,
        severity=ValidationSeverity.CRITICAL,
        message="grain uniqueness failed",
    )

    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_column_presence",
        return_value=ValidationResult(rule_name="column_presence", passed=True),
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_column_datatype",
        return_value=ValidationResult(rule_name="column_datatype", passed=True),
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_nullability",
        return_value=[],
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_allowed_values",
        return_value=[],
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_range",
        return_value=[],
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_array_lengths",
        return_value=[],
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset._run_uniqueness_validators",
        return_value=[critical_result],
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset._apply_thresholds",
        side_effect=lambda results, **kwargs: results,
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_foreign_keys",
        return_value=[],
    )
    mocker.patch(
        "instacart_etl_rnn.validation.dataset.validate_row_logic",
        return_value=[],
    )

    contract = {
        "dataset": {"name": "orders"},
        "schema": [{"name": "order_id"}],
    }

    with caplog.at_level("CRITICAL"):
        with pytest.raises(DataValidationError):
            validate_dataset(
                df,
                contract=contract,
            )

    assert "grain uniqueness failed" in caplog.text


def test_validate_dataset_detects_duplicate_order_id(spark):
    df = spark.createDataFrame(
        [
            (1, 10, "prior"),
            (1, 20, "train"),
        ],
        ["order_id", "user_id", "eval_set"],
    )

    contract = {
        "dataset": {
            "name": "orders",
            "grain": ["order_id"],
        },
        "schema": [
            {
                "name": "order_id",
                "type": "integer",
                "nullable": False,
                "constraints": {"unique": True, "minimum": 1},
                "thresholds": {
                    "uniqueness": {
                        "max_failed_percent": 0,
                        "severity": "critical",
                    },
                },
            },
            {
                "name": "user_id",
                "type": "integer",
                "nullable": False,
            },
            {
                "name": "eval_set",
                "type": "string",
                "nullable": False,
                "constraints": {"allowed_values": ["prior", "train", "test"]},
            },
        ],
    }

    with pytest.raises(DataValidationError) as exc_info:
        validate_dataset(
            df,
            contract=contract,
        )

    uniqueness_results = [
        result
        for result in exc_info.value.report.results
        if "uniqueness" in result.rule_name
    ]

    assert uniqueness_results
    assert any(result.failed_count > 0 for result in uniqueness_results)
