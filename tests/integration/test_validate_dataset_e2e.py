import logging
from copy import deepcopy

import pytest
from pyspark.sql import functions as F

from instacart_etl_rnn.validation.dataset import validate_dataset
from instacart_etl_rnn.validation.exceptions import DataValidationError
from instacart_etl_rnn.validation.models import (
    ValidationSeverity,
    ValidationStatus,
)
from tests.helpers import find_result


def test_validate_dataset_accepts_valid_orders(
    validate_dataset_orders_df,
    validate_dataset_orders_contract,
):
    report = validate_dataset(
        validate_dataset_orders_df,
        contract=validate_dataset_orders_contract,
    )

    assert report.dataset_name == "orders"
    assert report.has_errors is False

    assert all(result.failed_count == 0 for result in report.results)


def test_validate_dataset_fails_presence_gate(
    validate_dataset_orders_df,
    validate_dataset_orders_contract,
    caplog,
):
    bad_df = validate_dataset_orders_df.drop("user_id")

    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(DataValidationError) as exc_info:
            validate_dataset(
                bad_df,
                contract=validate_dataset_orders_contract,
            )

    report = exc_info.value.report

    assert report.dataset_name == "orders"
    assert report.results == []

    assert len(caplog.records) == 1

    record = caplog.records[0]

    assert record.levelno == logging.CRITICAL
    assert "missing columns: user_id;" in record.message


def test_validate_dataset_fails_datatype_gate(
    validate_dataset_orders_df,
    validate_dataset_orders_contract,
    caplog,
):
    bad_df = validate_dataset_orders_df.withColumn(
        "order_id",
        F.col("order_id").cast("string"),
    )

    with caplog.at_level(logging.CRITICAL):
        with pytest.raises(DataValidationError) as exc_info:
            validate_dataset(
                bad_df,
                contract=validate_dataset_orders_contract,
            )

    report = exc_info.value.report

    assert report.dataset_name == "orders"
    assert report.results == []

    assert len(caplog.records) == 1

    record = caplog.records[0]

    assert record.levelno == logging.CRITICAL
    assert "Incompatible column data types" in record.message


def test_validate_dataset_rejects_range_failure(
    validate_dataset_orders_df,
    validate_dataset_orders_contract,
):
    bad_df = validate_dataset_orders_df.withColumn(
        "order_number",
        F.when(
            F.col("order_id") == 2,
            F.lit(0),
        ).otherwise(F.col("order_number")),
    )

    with pytest.raises(DataValidationError) as exc_info:
        validate_dataset(
            bad_df,
            contract=validate_dataset_orders_contract,
        )

    result = find_result(
        exc_info.value.report,
        rule_name="order_number.range",
    )

    assert result.failed_count == 1
    assert result.status == ValidationStatus.FAILED
    assert result.severity == ValidationSeverity.ERROR
    assert result.passed is False
    assert "order_number.range failed: 1 row(s) violated the rule" in result.message


def test_validate_dataset_returns_warning_within_threshold(
    validate_dataset_orders_df,
    validate_dataset_orders_contract,
):
    contract = deepcopy(validate_dataset_orders_contract)

    order_dow_schema = next(
        column for column in contract["schema"] if column["name"] == "order_dow"
    )

    order_dow_schema["thresholds"] = {
        "range": {
            "max_failed_percent": 20.0,
            "severity": "error",
        }
    }

    bad_df = validate_dataset_orders_df.withColumn(
        "order_dow",
        F.when(
            F.col("order_id") == 2,
            F.lit(7),
        ).otherwise(F.col("order_dow")),
    )

    report = validate_dataset(
        bad_df,
        contract=contract,
    )

    result = find_result(
        report,
        rule_name="order_dow.range",
    )

    assert result.failed_count == 1
    assert result.metadata["failed_percent"] == 20.0
    assert result.status == ValidationStatus.WARNING
    assert result.severity == ValidationSeverity.WARNING
    assert result.passed is True
    assert "order_dow.range produced a warning" in result.message

    assert report.has_errors is False


def test_validate_dataset_rejects_duplicate_order_id(
    validate_dataset_orders_df,
    validate_dataset_orders_contract,
):
    duplicate = validate_dataset_orders_df.filter(F.col("order_id") == 1)

    bad_df = validate_dataset_orders_df.unionByName(duplicate)

    with pytest.raises(DataValidationError) as exc_info:
        validate_dataset(
            bad_df,
            contract=validate_dataset_orders_contract,
        )

    result = find_result(
        exc_info.value.report,
        rule_name="order_id.uniqueness",
    )

    assert result.failed_count > 0
    assert result.status == ValidationStatus.FAILED
    assert result.severity == ValidationSeverity.CRITICAL
    assert result.passed is False
    assert "Columns 'order_id'" in result.message
    assert "row(s) participating in duplicate key values" in result.message


def test_validate_dataset_rejects_row_logic_failure(
    validate_dataset_orders_df,
    validate_dataset_orders_contract,
):
    bad_df = validate_dataset_orders_df.withColumn(
        "days_since_prior_order",
        F.when(
            F.col("order_number") == 1,
            F.lit(5.0),
        ).otherwise(F.col("days_since_prior_order")),
    )

    with pytest.raises(DataValidationError) as exc_info:
        validate_dataset(
            bad_df,
            contract=validate_dataset_orders_contract,
        )

    result = find_result(
        exc_info.value.report,
        rule_name="first_order_has_no_prior_interval",
    )

    assert result.failed_count == 2
    assert result.status == ValidationStatus.FAILED
    assert result.severity == ValidationSeverity.CRITICAL
    assert result.passed is False
