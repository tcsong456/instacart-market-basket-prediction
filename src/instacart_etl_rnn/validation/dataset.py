import logging
from dataclasses import replace
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from instacart_etl_rnn.validation.allowed_values import allowed_values_validator
from instacart_etl_rnn.validation.array_length import array_length_validator
from instacart_etl_rnn.validation.exceptions import (
    DataValidationError,
    EmptyDatasetError,
    InvalidConstraintError,
    InvalidContractError,
)
from instacart_etl_rnn.validation.models import (
    ThresholdConfig,
    ValidationMetric,
    ValidationReport,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)
from instacart_etl_rnn.validation.nullability import nullability_validator
from instacart_etl_rnn.validation.pattern import pattern_validator
from instacart_etl_rnn.validation.range import range_validator
from instacart_etl_rnn.validation.referential_integrity import validate_foreign_keys
from instacart_etl_rnn.validation.row_logic import validate_row_logic
from instacart_etl_rnn.validation.schema import (
    validate_column_datatype,
    validate_column_presence,
)
from instacart_etl_rnn.validation.string_length import string_length_validator
from instacart_etl_rnn.validation.uniqueness import validate_uniqueness
from instacart_etl_rnn.validation.utils import (
    is_non_empty_string_list,
    is_non_negative_number,
)

logger = logging.getLogger(__name__)


def _get_metric_threshold(
    metric: ValidationMetric,
    *,
    contract: dict[str, Any],
) -> ThresholdConfig:
    column_name = metric.columns[0]

    column_schema = next(
        schema for schema in contract["schema"] if schema["name"] == column_name
    )

    threshold_config = column_schema.get("thresholds", {}).get(
        metric.validation_type, {}
    )

    maximum_failed_percent = threshold_config.get(
        "max_failed_percent",
        0.0,
    )

    severity_value = threshold_config.get(
        "severity",
        "error",
    )

    if not is_non_negative_number(maximum_failed_percent):
        raise InvalidConstraintError("max_failed_percent must be a non-negative number")

    if severity_value == "error":
        severity = ValidationSeverity.ERROR
    elif severity_value == "critical":
        severity = ValidationSeverity.CRITICAL
    else:
        raise InvalidConstraintError("severity must be either 'error' or 'critical'")

    return ThresholdConfig(
        maximum_failed_percent=maximum_failed_percent,
        severity=severity,
    )


def _evaluate_metric(
    metric: ValidationMetric,
    *,
    failed_count: int,
    total_rows: int,
    contract: dict[str, Any],
) -> ValidationResult:
    threshold = _get_metric_threshold(
        metric,
        contract=contract,
    )

    failed_percent = failed_count / total_rows * 100

    if failed_percent == 0:
        status = ValidationStatus.PASSED
        severity = ValidationSeverity.INFO

        message = f"{metric.rule_name} passed"
    elif failed_percent <= threshold.maximum_failed_percent:
        status = ValidationStatus.WARNING
        severity = ValidationSeverity.WARNING

        message = (
            f"{metric.rule_name} produced a warning: "
            f"{failed_count} row(s) violated the rule "
            f"({failed_percent:.2f}%)"
        )
    else:
        status = ValidationStatus.FAILED
        severity = threshold.severity

        message = (
            f"{metric.rule_name} failed: "
            f"{failed_count} row(s) violated the rule "
            f"({failed_percent:.2f}%), exceeding the "
            f"maximum allowed failed percent"
        )

    return ValidationResult(
        rule_name=metric.rule_name,
        category=metric.validation_type,
        passed=status in (ValidationStatus.PASSED, ValidationStatus.WARNING),
        status=status,
        severity=severity,
        failed_count=failed_count,
        invalid_rows=None,
        message=message,
        metadata={
            "columns": metric.columns,
            "failed_percent": failed_percent,
            "maximum_failed_percent": (threshold.maximum_failed_percent),
        },
    )


def _build_column_metrics(
    contract: dict[str, Any],
) -> list[ValidationMetric]:
    metrics: list[ValidationMetric] = []

    metrics.extend(nullability_validator(contract))

    metrics.extend(range_validator(contract))

    metrics.extend(allowed_values_validator(contract))

    metrics.extend(string_length_validator(contract))

    metrics.extend(pattern_validator(contract))

    metrics.extend(array_length_validator(contract))

    return metrics


def _run_column_metrics(
    df: DataFrame,
    *,
    contract: dict[str, Any],
) -> tuple[list[ValidationResult], int]:
    metrics = _build_column_metrics(contract)

    aggregations = [
        F.count("*").alias("total_rows"),
        *[metric.expression for metric in metrics],
    ]

    aggregated = df.agg(*aggregations).first()

    total_rows = int(aggregated["total_rows"])

    if total_rows == 0:
        raise EmptyDatasetError("DataFrame must not be empty")

    results = []

    for metric in metrics:
        failed_count = int(aggregated[metric.alias] or 0)

        results.append(
            _evaluate_metric(
                metric,
                failed_count=failed_count,
                total_rows=total_rows,
                contract=contract,
            )
        )

    return results, total_rows


def _apply_strict_result_status(
    result: ValidationResult,
    *,
    total_rows: int,
    severity: ValidationSeverity = (ValidationSeverity.CRITICAL),
) -> ValidationResult:
    failed_percent = result.failed_count / total_rows * 100

    metadata = {
        **result.metadata,
        "failed_percent": failed_percent,
    }

    if result.failed_count == 0:
        return replace(
            result,
            passed=True,
            status=ValidationStatus.PASSED,
            severity=ValidationSeverity.INFO,
            metadata=metadata,
        )

    return replace(
        result,
        passed=False,
        status=ValidationStatus.FAILED,
        severity=severity,
        metadata=metadata,
    )


def _run_uniqueness_validators(
    df: DataFrame,
    *,
    contract: dict[str, Any],
) -> list[ValidationResult]:
    results: list[ValidationResult] = []
    checked_column_sets: set[tuple[str, ...]] = set()

    for column_schema in contract.get(
        "schema",
        [],
    ):
        unique = column_schema.get("constraints", {}).get("unique", False)

        if not isinstance(unique, bool):
            raise InvalidContractError(
                f"'unique' for column {column_schema.get('name')!r} must be a boolean"
            )

        if not unique:
            continue

        columns = (column_schema["name"],)

        results.append(
            validate_uniqueness(
                df,
                columns=list(columns),
            )
        )

        checked_column_sets.add(columns)

    grain = contract.get("dataset", {}).get("grain")

    if grain is not None:
        if not is_non_empty_string_list(grain):
            raise InvalidContractError(
                "The dataset grain must be a non-empty list of column names"
            )

        grain_key = tuple(grain)

        if grain_key not in checked_column_sets:
            results.append(
                validate_uniqueness(
                    df,
                    columns=grain,
                )
            )

    return results


def _log_validation_results(
    results: list[ValidationResult],
) -> None:
    for result in results:
        if result.status == ValidationStatus.PASSED:
            continue

        match result.severity:
            case ValidationSeverity.WARNING:
                logger.warning(result.message)

            case ValidationSeverity.ERROR:
                logger.error(result.message)

            case ValidationSeverity.CRITICAL:
                logger.critical(result.message)


def validate_dataset(
    df: DataFrame,
    *,
    contract: dict[str, Any],
    reference_datasets: dict[str, DataFrame] | None = None,
) -> ValidationReport:
    reference_datasets = reference_datasets or {}

    dataset_name = str(contract.get("dataset", {}).get("name", "unknown"))

    results: list[ValidationResult] = []

    presence_result = validate_column_presence(
        df,
        contract=contract,
    )

    if not presence_result.passed:
        logger.critical(presence_result.message)

        raise DataValidationError(
            ValidationReport(
                dataset_name=dataset_name,
                results=results,
            )
        )

    datatype_result = validate_column_datatype(
        df,
        contract=contract,
    )

    if not datatype_result.passed:
        logger.critical(datatype_result.message)

        raise DataValidationError(
            ValidationReport(
                dataset_name=dataset_name,
                results=results,
            )
        )

    column_results, total_rows = _run_column_metrics(
        df,
        contract=contract,
    )

    results.extend(column_results)

    uniqueness_results = _run_uniqueness_validators(
        df,
        contract=contract,
    )

    uniqueness_results = [
        _apply_strict_result_status(
            result,
            total_rows=total_rows,
            severity=ValidationSeverity.CRITICAL,
        )
        for result in uniqueness_results
    ]

    results.extend(uniqueness_results)

    ri_results = validate_foreign_keys(
        child_df=df,
        contract=contract,
        datasets=reference_datasets,
    )

    ri_results = [
        _apply_strict_result_status(
            result,
            total_rows=total_rows,
            severity=ValidationSeverity.CRITICAL,
        )
        for result in ri_results
    ]

    results.extend(ri_results)

    row_logic_results = validate_row_logic(
        df,
        contract=contract,
    )

    row_logic_results = [
        _apply_strict_result_status(
            result,
            total_rows=total_rows,
            severity=ValidationSeverity.CRITICAL,
        )
        for result in row_logic_results
    ]

    results.extend(row_logic_results)

    report = ValidationReport(
        dataset_name=dataset_name,
        results=results,
    )

    _log_validation_results(results)

    if report.has_errors:
        raise DataValidationError(report)

    logger.info(
        "Dataset %s passed data validation",
        dataset_name,
    )

    return report
