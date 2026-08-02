import logging
from typing import Any

from pyspark.sql import DataFrame

from instacart_etl_rnn.validation.allowed_values import validate_allowed_values
from instacart_etl_rnn.validation.array_length import validate_array_lengths
from instacart_etl_rnn.validation.exceptions import (
    DataValidationError,
    EmptyDatasetError,
    InvalidConstraintError,
    InvalidContractError,
)
from instacart_etl_rnn.validation.models import (
    ValidationReport,
    ValidationResult,
    ValidationSeverity,
    ValidationStatus,
)
from instacart_etl_rnn.validation.null import validate_nullability
from instacart_etl_rnn.validation.range import validate_range
from instacart_etl_rnn.validation.referential_integrity import validate_foreign_keys
from instacart_etl_rnn.validation.row_logic import validate_row_logic
from instacart_etl_rnn.validation.schema import (
    validate_column_datatype,
    validate_column_presence,
)
from instacart_etl_rnn.validation.uniqueness import validate_uniqueness
from instacart_etl_rnn.validation.utils import is_non_negative_number

logger = logging.getLogger(__name__)


VALIDATION_TYPES = [
    "nullability",
    "range",
    "allowed_values",
    "array_length",
    "uniqueness",
]


def _run_uniqueness_validators(
    df: DataFrame, *, contract: dict[str, Any]
) -> list[ValidationResult]:
    results = []
    checked_column_sets = set()

    schemas = contract.get("schema")

    for column_schema in schemas:
        unique = column_schema.get("constraints", {}).get("unique", False)

        if not isinstance(unique, bool):
            raise InvalidContractError(
                f"'unique' for column {column_schema.get('name')!r} must be a boolean"
            )

        if not unique:
            continue

        column_name = column_schema["name"]
        columns = [column_name]
        results.append(validate_uniqueness(df, columns=columns))

        checked_column_sets.add(tuple(columns))

    grain = contract.get("dataset", {}).get("grain")
    if grain is not None:
        if (
            not isinstance(grain, list)
            or not grain
            or not all([isinstance(c, str) for c in grain])
        ):
            raise InvalidContractError(
                "The dataset grain must be a non-empty list of column names"
            )

        if tuple(grain) not in checked_column_sets:
            results.append(validate_uniqueness(df, columns=grain))

    return results


def _apply_thresholds(
    results: list[ValidationResult],
    validation_type: str,
    total_rows: int,
    *,
    contract: dict[str, Any],
) -> list[ValidationResult]:
    if validation_type not in VALIDATION_TYPES:
        raise ValueError(f"validation type: {validation_type} is not supported")

    schema_by_name = {schema["name"]: schema for schema in contract.get("schema")}

    for result in results:
        if result.failed_count == 0:
            continue

        result.status = ValidationStatus.FAILED
        result.severity = ValidationSeverity.ERROR

        failed_percent = result.failed_count / total_rows * 100

        result.message = (
            f"{result.rule_name} failed {result.failed_count} rows, "
            f"{failed_percent:.2f}% of total rows"
        )

        result_column = result.rule_name.split(".", 1)[0]
        result_columns = result_column.split(",")
        if len(result_columns) >= 2:
            result.severity = ValidationSeverity.CRITICAL
            continue

        column_schema = schema_by_name.get(result_column)
        if column_schema is None:
            result.message += "; no matching column threshold was found"
            continue

        thresholds = column_schema.get("thresholds", {}).get(validation_type, {})
        max_failed_percent = thresholds.get("max_failed_percent")
        severity = thresholds.get("severity", "error")

        if max_failed_percent is not None and not is_non_negative_number(
            max_failed_percent
        ):
            raise InvalidConstraintError(
                "max_failed_percent must be a non-negative number"
            )

        if severity not in {"error", "critical"}:
            raise InvalidConstraintError(
                "severity must be either 'error' or 'critical'"
            )

        if severity == "critical":
            result.severity = ValidationSeverity.CRITICAL

        if max_failed_percent is not None and failed_percent <= max_failed_percent:
            result.status = ValidationStatus.WARNING
            result.severity = ValidationSeverity.WARNING
            result.message += ", but did not exceed the maximum allowable percent"

    return results


def _apply_strict_thresholds(
    results: list[ValidationResult], total_rows: int, prefix: str = ""
) -> list[ValidationResult]:
    for result in results:
        failed_count = result.failed_count
        if failed_count == 0:
            continue

        failed_percent = failed_count / total_rows * 100
        if failed_percent > 0:
            result.message = (
                f"{prefix} rule: {result.rule_name} "
                f"failed {result.failed_count} rows, "
                f"{failed_percent:.2f}% of total rows"
            )
            result.severity = ValidationSeverity.CRITICAL
            result.status = ValidationStatus.FAILED

    return results


def validate_dataset(
    df: DataFrame,
    *,
    contract: dict[str, Any],
    reference_datasets: dict[str, DataFrame] | None = None,
) -> ValidationReport:
    if df.isEmpty():
        raise EmptyDatasetError("dataframe must not be empty")

    reference_datasets = reference_datasets or {}

    dataset_name = str(contract.get("dataset", {}).get("name", "unknown"))

    results = []
    total_rows = df.count()

    column_presence_result = validate_column_presence(df, contract=contract)
    results.append(column_presence_result)
    if not column_presence_result.passed:
        column_presence_result.status = ValidationStatus.FAILED
        column_presence_result.severity = ValidationSeverity.CRITICAL
        logger.critical(column_presence_result.message)
        raise DataValidationError(
            ValidationReport(dataset_name=dataset_name, results=results)
        )

    column_datatype_result = validate_column_datatype(df, contract=contract)
    results.append(column_datatype_result)
    if not column_datatype_result.passed:
        column_datatype_result.status = ValidationStatus.FAILED
        column_datatype_result.severity = ValidationSeverity.CRITICAL
        logger.critical(column_datatype_result.message)
        raise DataValidationError(
            ValidationReport(dataset_name=dataset_name, results=results)
        )

    results.extend(
        _apply_thresholds(
            results=validate_nullability(df, contract=contract),
            total_rows=total_rows,
            validation_type="nullability",
            contract=contract,
        )
    )

    results.extend(
        _apply_thresholds(
            results=validate_allowed_values(df, contract=contract),
            total_rows=total_rows,
            validation_type="allowed_values",
            contract=contract,
        )
    )

    results.extend(
        _apply_thresholds(
            results=validate_range(df, contract=contract),
            total_rows=total_rows,
            validation_type="range",
            contract=contract,
        )
    )

    results.extend(
        _apply_thresholds(
            results=validate_array_lengths(df, contract=contract),
            total_rows=total_rows,
            validation_type="array_length",
            contract=contract,
        )
    )

    results.extend(
        _apply_thresholds(
            results=_run_uniqueness_validators(df, contract=contract),
            total_rows=total_rows,
            validation_type="uniqueness",
            contract=contract,
        )
    )

    results.extend(
        _apply_strict_thresholds(
            results=validate_foreign_keys(
                child_df=df, datasets=reference_datasets, contract=contract
            ),
            total_rows=total_rows,
            prefix="RI",
        )
    )

    results.extend(
        _apply_strict_thresholds(
            results=validate_row_logic(df, contract=contract),
            total_rows=total_rows,
            prefix="business",
        )
    )

    report = ValidationReport(dataset_name=dataset_name, results=results)

    for result in results:
        if result.failed_count == 0:
            continue

        message = result.message + "\n"
        match result.severity:
            case ValidationSeverity.WARNING:
                logger.warning(message)
            case ValidationSeverity.ERROR:
                logger.error(message)
            case ValidationSeverity.CRITICAL:
                logger.critical(message)

    if report.has_errors:
        raise DataValidationError(report)
    elif report.can_continue:
        logger.info(
            "All column and contract rules passed, "
            "Dataset: %s has passed the data validation",
            dataset_name,
        )
        return report
