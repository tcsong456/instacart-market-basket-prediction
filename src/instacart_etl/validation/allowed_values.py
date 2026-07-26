from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from instacart_etl.validation.exceptions import InvalidConstraintError
from instacart_etl.validation.models import ValidationResult


def _build_invalid_conditions(column_name: str, allowed_values: list[Any]):
    return F.col(column_name).isNotNull() & ~F.col(column_name).isin(allowed_values)


def validate_allowed_values(
    df: DataFrame, *, contract: dict[str, Any]
) -> list[ValidationResult]:
    schemas = contract.get("schema")

    allowed_values_columns, allowed_values_list = [], []
    for schema in schemas:
        column_name = schema["name"]
        constraints = schema.get("constraints")

        if constraints is not None:
            allowed_values = constraints.get("allowed_values")
            if allowed_values is not None:
                if not isinstance(allowed_values, list):
                    raise InvalidConstraintError(
                        "allowed_values in constraint must be a list"
                    )

                if not allowed_values:
                    raise InvalidConstraintError(
                        "allowed_values must not be an empty list"
                    )

                allowed_values_columns.append(column_name)
                allowed_values_list.append(allowed_values)

    if allowed_values_columns:
        failed_count = (
            df.agg(
                *[
                    F.sum(
                        F.when(
                            _build_invalid_conditions(column, allowed_values), 1
                        ).otherwise(0)
                    ).alias(column)
                    for column, allowed_values in zip(
                        allowed_values_columns, allowed_values_list
                    )
                ]
            ).first()
            or 0
        )
        failed_dict = {
            column: int(failed_count[column] or 0) for column in allowed_values_columns
        }
    else:
        failed_dict = {}

    results = []
    if not failed_dict:
        results = []
    else:
        for (key, value), allowed_values in zip(
            failed_dict.items(), allowed_values_list
        ):
            if value > 0:
                invalid_condition = _build_invalid_conditions(
                    column_name=key, allowed_values=allowed_values
                )
                invalid_rows = (
                    df.filter(invalid_condition).select(key).distinct().limit(20)
                )
                results.append(
                    ValidationResult(
                        rule_name=f"{key}.allowed_values",
                        category="allowed_values",
                        passed=False,
                        failed_count=value,
                        invalid_rows=invalid_rows,
                        message=(
                            f"Column '{key}' contains {value} "
                            "row(s) with disallowed values"
                        ),
                        metadata={"column_name": key, "allowed_values": allowed_values},
                    )
                )
            else:
                results.append(
                    ValidationResult(
                        rule_name=f"{key}.allowed_values",
                        category="allowed_values",
                        passed=True,
                        failed_count=0,
                        invalid_rows=None,
                        message=f"Column: all values of '{key}' are allowed",
                        metadata={"column_name": key, "allowed_values": allowed_values},
                    )
                )

    return results
