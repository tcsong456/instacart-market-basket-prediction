from typing import Any, Sequence

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from instacart_etl.validation.models import ValidationResult


def validate_allowed_values(
    df: DataFrame, *, column_name: str, allowed_values: Sequence[Any]
) -> ValidationResult:
    if not allowed_values:
        raise ValueError("allowed_values can not be empty")

    invalid_conditions = F.col(column_name).isNotNull() & ~F.col(column_name).isin(
        list(allowed_values)
    )

    failed_count = (
        df.agg(
            F.sum(F.when(invalid_conditions, F.lit(1)).otherwise(F.lit(0))).alias(
                "failed_count"
            )
        ).first()["failed_count"]
        or 0
    )

    failed_count = int(failed_count)
    passed = failed_count == 0

    if not passed:
        invalid_rows = (
            df.filter(invalid_conditions).select(column_name).distinct().limit(20)
        )
    else:
        invalid_rows = None

    message = (
        f"Column: all values of '{column_name}' are allowed"
        if passed
        else (
            f"Column '{column_name}' contains {failed_count} "
            "row(s) with disallowed values"
        )
    )

    return ValidationResult(
        rule_name=f"{column_name}.allowed_values",
        category="allowed_values",
        passed=passed,
        failed_count=failed_count,
        invalid_rows=invalid_rows,
        message=message,
        metadata={"column_name": column_name, "allowed_values": list(allowed_values)},
    )
