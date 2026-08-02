from copy import deepcopy

import pytest
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from instacart_etl_rnn.validation.dataset import validate_dataset
from instacart_etl_rnn.validation.exceptions import (
    DataValidationError,
    EmptyDatasetError,
)
from instacart_etl_rnn.validation.models import ValidationSeverity, ValidationStatus
from tests.helpers import find_result


def test_validate_dataset_e2e_valid_dataset_passes(
    validate_dataset_orders_df,
    validate_dataset_users_df,
    validate_dataset_orders_contract,
):
    report = validate_dataset(
        contract=validate_dataset_orders_contract,
        reference_datasets={"users": validate_dataset_users_df},
        df=validate_dataset_orders_df,
    )

    assert report.dataset_name == "orders"
    assert report.results
    assert report.has_errors is False

    assert all(result.failed_count == 0 for result in report.results)

    assert all(result.status == ValidationStatus.PASSED for result in report.results)


def test_validate_dataset_e2e_empty_dataset_raises(
    spark,
    validate_dataset_orders_schema,
    validate_dataset_users_df,
    validate_dataset_orders_contract,
):
    empty_orders_df = spark.createDataFrame(
        [],
        schema=validate_dataset_orders_schema,
    )

    with pytest.raises(EmptyDatasetError):
        validate_dataset(
            empty_orders_df,
            contract=validate_dataset_orders_contract,
            reference_datasets={"users": validate_dataset_users_df},
        )


def test_validate_dataset_e2e_missing_column_fails_fast(
    validate_dataset_orders_df,
    validate_dataset_users_df,
    validate_dataset_orders_contract,
):
    invalid_df = validate_dataset_orders_df.drop("eval_set")
    with pytest.raises(DataValidationError) as exc_info:
        validate_dataset(
            invalid_df,
            contract=validate_dataset_orders_contract,
            reference_datasets={"users": validate_dataset_users_df},
        )

    report = exc_info.value.report

    presence_result = find_result(report, rule_name="column_presence")

    assert presence_result.failed_count == 1
    assert presence_result.status == ValidationStatus.FAILED
    assert presence_result.severity == ValidationSeverity.CRITICAL
    assert report.results == [presence_result]


def test_validate_dataset_e2e_wrong_datatype_fails_fast(
    spark,
    validate_dataset_users_df,
    validate_dataset_orders_contract,
):
    wrong_schema = StructType(
        [
            StructField("order_id", IntegerType(), False),
            StructField("user_id", IntegerType(), False),
            StructField("eval_set", IntegerType(), False),
            StructField("order_number", IntegerType(), False),
            StructField(
                "days_since_prior_order",
                DoubleType(),
                True,
            ),
        ]
    )

    invalid_df = spark.createDataFrame(
        [
            (1, 1, 100, 1, None),
        ],
        schema=wrong_schema,
    )

    with pytest.raises(DataValidationError) as exc_info:
        validate_dataset(
            invalid_df,
            contract=validate_dataset_orders_contract,
            reference_datasets={"users": validate_dataset_users_df},
        )

    report = exc_info.value.report
    datatype_result = find_result(report, rule_name="column_types")
    presence_result = find_result(report, rule_name="column_presence")

    assert datatype_result.failed_count == 1
    assert datatype_result.status == ValidationStatus.FAILED
    assert datatype_result.severity == ValidationSeverity.CRITICAL
    assert report.results == [presence_result, datatype_result]


def test_validate_dataset_e2e_non_nullable_column_contains_null(
    spark,
    validate_dataset_users_df,
    validate_dataset_orders_contract,
):
    nullable_schema = StructType(
        [
            StructField("order_id", IntegerType(), True),
            StructField("user_id", IntegerType(), True),
            StructField("eval_set", StringType(), True),
            StructField("order_number", IntegerType(), True),
            StructField(
                "days_since_prior_order",
                DoubleType(),
                True,
            ),
        ]
    )

    invalid_df = spark.createDataFrame(
        [
            (1, 1, "prior", 1, None),
            (2, 1, None, 2, 5.0),
        ],
        schema=nullable_schema,
    )

    with pytest.raises(DataValidationError) as exc_info:
        validate_dataset(
            invalid_df,
            contract=validate_dataset_orders_contract,
            reference_datasets={"users": validate_dataset_users_df},
        )

    result = find_result(
        exc_info.value.report,
        rule_name="eval_set.nullability",
    )

    assert result.failed_count == 1
    assert result.status == ValidationStatus.FAILED
    assert result.severity == ValidationSeverity.ERROR


def test_validate_dataset_e2e_disallowed_value_fails(
    spark,
    validate_dataset_orders_schema,
    validate_dataset_users_df,
    validate_dataset_orders_contract,
):
    invalid_df = spark.createDataFrame(
        [
            (1, 1, "prior", 1, None),
            (2, 1, "invalid", 2, 5.0),
        ],
        schema=validate_dataset_orders_schema,
    )

    with pytest.raises(DataValidationError) as exc_info:
        validate_dataset(
            invalid_df,
            contract=validate_dataset_orders_contract,
            reference_datasets={"users": validate_dataset_users_df},
        )

    result = find_result(
        exc_info.value.report,
        rule_name="allowed_values",
    )

    assert result.failed_count == 1
    assert result.status == ValidationStatus.FAILED


def test_validate_dataset_e2e_range_violation_fails(
    spark,
    validate_dataset_orders_schema,
    validate_dataset_users_df,
    validate_dataset_orders_contract,
):
    invalid_df = spark.createDataFrame(
        [
            (1, 1, "prior", 1, None),
            (2, 1, "train", 0, 5.0),
        ],
        schema=validate_dataset_orders_schema,
    )

    with pytest.raises(DataValidationError) as exc_info:
        validate_dataset(
            invalid_df,
            contract=validate_dataset_orders_contract,
            reference_datasets={"users": validate_dataset_users_df},
        )

    result = find_result(
        exc_info.value.report,
        rule_name="order_number.range",
    )

    assert result.failed_count == 1
    assert result.status == ValidationStatus.FAILED


def test_validate_dataset_e2e_duplicate_grain_fails(
    spark,
    validate_dataset_orders_schema,
    validate_dataset_users_df,
    validate_dataset_orders_contract,
):
    invalid_df = spark.createDataFrame(
        [
            (1, 1, "prior", 1, None),
            (1, 1, "train", 2, 5.0),
        ],
        schema=validate_dataset_orders_schema,
    )

    with pytest.raises(DataValidationError) as exc_info:
        validate_dataset(
            invalid_df,
            contract=validate_dataset_orders_contract,
            reference_datasets={"users": validate_dataset_users_df},
        )

    result = find_result(
        exc_info.value.report,
        rule_name="uniqueness",
    )

    assert result.failed_count == 2
    assert result.status == ValidationStatus.FAILED
    assert result.severity == ValidationSeverity.CRITICAL


def test_validate_dataset_e2e_missing_parent_key_fails(
    spark,
    validate_dataset_orders_schema,
    validate_dataset_users_df,
    validate_dataset_orders_contract,
):
    invalid_df = spark.createDataFrame(
        [
            (1, 1, "train", 1, None),
            (2, 999, "test", 1, None),
        ],
        schema=validate_dataset_orders_schema,
    )

    with pytest.raises(DataValidationError) as exc_info:
        validate_dataset(
            invalid_df,
            contract=validate_dataset_orders_contract,
            reference_datasets={"users": validate_dataset_users_df},
        )

    result = find_result(
        exc_info.value.report,
        rule_name="orders_user_fk",
    )

    assert result.failed_count == 1
    assert result.status == ValidationStatus.FAILED
    assert result.severity == ValidationSeverity.CRITICAL


def test_validate_dataset_e2e_business_rule_fails(
    spark,
    validate_dataset_orders_schema,
    validate_dataset_users_df,
    validate_dataset_orders_contract,
):
    invalid_df = spark.createDataFrame(
        [
            (1, 1, "prior", 1, None),
            (2, 1, "prior", 2, 5.0),
        ],
        schema=validate_dataset_orders_schema,
    )

    with pytest.raises(DataValidationError) as exc_info:
        validate_dataset(
            invalid_df,
            contract=validate_dataset_orders_contract,
            reference_datasets={"users": validate_dataset_users_df},
        )

    report = exc_info.value.report

    result = find_result(
        report,
        rule_name="last_order_is_train_or_test",
    )

    assert result.failed_count == 1
    assert result.status == ValidationStatus.FAILED
    assert result.severity == ValidationSeverity.CRITICAL

    result = find_result(
        report,
        rule_name="exactly_one_train_or_test_per_user",
    )

    assert result.failed_count == 2
    assert result.status == ValidationStatus.FAILED
    assert result.severity == ValidationSeverity.CRITICAL


def test_validate_dataset_e2e_soft_threshold_returns_warning(
    spark,
    validate_dataset_orders_schema,
    validate_dataset_users_df,
    validate_dataset_orders_contract,
):
    warning_contract = deepcopy(validate_dataset_orders_contract)
    for schema in warning_contract["schema"]:
        if schema["name"] == "order_number":
            schema["thresholds"] = {"range": {"max_failed_percent": 25.0}}
            break

    invalid_df = spark.createDataFrame(
        [
            (1, 1, "prior", 1, None),
            (2, 1, "train", 2, 5.0),
            (3, 2, "prior", 0, None),
            (4, 2, "test", 2, 4.0),
        ],
        schema=validate_dataset_orders_schema,
    )

    report = validate_dataset(
        invalid_df,
        contract=warning_contract,
        reference_datasets={"users": validate_dataset_users_df},
    )

    assert report.has_errors is False

    range_result = find_result(
        report,
        rule_name="range",
    )

    assert range_result.failed_count == 1
    assert range_result.status == ValidationStatus.WARNING
    assert range_result.severity == ValidationSeverity.WARNING
