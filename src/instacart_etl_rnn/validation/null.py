from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from instacart_etl_rnn.validation.exceptions import InvalidConstraintError
from instacart_etl_rnn.validation.models import ValidationResult


def validate_nullability(
    df: DataFrame, *, contract: dict[str, Any]
) -> list[ValidationResult]:
    schemas = contract.get("schema")

    non_null_columns = []

    for schema in schemas:
        column_name = schema["name"]
        column_nullability = schema["nullable"]

        if not isinstance(column_nullability, bool):
            raise InvalidConstraintError(
                f"Column '{column_name}' must have a boolean 'nullable' property"
            )

        if not column_nullability:
            non_null_columns.append(column_name)

    if non_null_columns:
        null_rows_count = df.agg(
            *[
                F.sum(
                    F.when(F.col(column).isNull(), F.lit(1)).otherwise(F.lit(0))
                ).alias(column)
                for column in non_null_columns
            ]
        ).first()

        null_counts = {
            column: int(null_rows_count[column]) for column in non_null_columns
        }

    results = []
    for schema in schemas:
        name = schema["name"]
        null = schema["nullable"]

        if null:
            results.append(
                ValidationResult(
                    rule_name=f"{name}.nullability",
                    category="nullability",
                    passed=True,
                    message=f"Column: '{name}' allows null values",
                    failed_count=0,
                    invalid_rows=None,
                    metadata={"column_name": name, "nullable": True},
                )
            )
        else:
            null_count = null_counts[name]
            passed = null_count == 0
            if not passed:
                invalid_rows = df.filter(F.col(name).isNull()).limit(20)
            else:
                invalid_rows = None

            message = (
                f"Column '{name}' contains no null values"
                if passed
                else (
                    f"Column '{name}' is not nullable but contains "
                    f"{null_count} null value(s)"
                )
            )

            results.append(
                ValidationResult(
                    rule_name=f"{name}.nullability",
                    category="nullability",
                    passed=passed,
                    failed_count=null_count,
                    invalid_rows=invalid_rows,
                    message=message,
                    metadata={"column_name": name, "nullable": False},
                )
            )

    return results
