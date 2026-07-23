from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from instacart_etl.validation.message import _build_range_message
from instacart_etl.validation.models import ValidationResult
from instacart_etl.validation.exceptions import InvalidConstraintError


def validate_range(
    df: DataFrame,
    *,
    column_name: str,
    min_value: int | float | None,
    max_value: int | float | None,
) -> ValidationResult:
    if min_value is None and max_value is None:
        raise InvalidConstraintError(
            "At least one of minimum value and maximum value must be provided"
        )

    if min_value is not None and max_value is not None and min_value > max_value:
        raise InvalidConstraintError(
            "minimum value should not be greater than maximum value"
            f"but {min_value} > {max_value}"
        )

    invalid_condition = F.lit(False)
    if min_value is not None:
        invalid_condition = invalid_condition | (F.col(column_name) < F.lit(min_value))

    if max_value is not None:
        invalid_condition = invalid_condition | (F.col(column_name) > F.lit(max_value))

    invalid_rows = df.filter(F.col(column_name).isNotNull() & invalid_condition)
    invalid_count = invalid_rows.count()

    sample_invalid_rows = invalid_rows.limit(30) if invalid_count else None

    return ValidationResult(
        rule_name=f"{column_name}.range",
        category="range",
        passed=(invalid_count == 0),
        message=_build_range_message(
            column_name=column_name, min_value=min_value, max_value=max_value
        ),
        failed_count=invalid_count,
        metadata={
            "column_name": column_name,
            "minimum": min_value,
            "maximum": max_value,
        },
        invalid_rows=sample_invalid_rows,
    )
