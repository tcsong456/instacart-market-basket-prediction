from typing import Any

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from instacart_etl_rnn.validation.exceptions import InvalidConstraintError
from instacart_etl_rnn.validation.models import ValidationResult
from instacart_etl_rnn.validation.utils import is_number


def _build_invalid_condition(
    column_name: str,
    minimum: int | float | None,
    maximum: int | float | None,
) -> Column:
    invalid_condition = F.lit(False)

    if minimum is not None:
        invalid_condition = invalid_condition | (F.col(column_name) < minimum)

    if maximum is not None:
        invalid_condition = invalid_condition | (F.col(column_name) > maximum)

    return invalid_condition & F.col(column_name).isNotNull()


def _build_range_message(
    column_name: str, min_value: int | float | None, max_value: int | float | None
) -> str:
    if min_value is not None and max_value is not None:
        return f"column '{column_name}' must be between {min_value} and {max_value}"

    if min_value is not None:
        return f"column '{column_name}' must be at least '{min_value}'"

    if max_value is not None:
        return f"column '{column_name}' must be at most '{max_value}'"


def validate_range(
    df: DataFrame, *, contract: dict[str, Any]
) -> list[ValidationResult]:
    schemas = contract.get("schema")

    range_columns, range_values = [], []
    for schema in schemas:
        name = schema["name"]
        constraints = schema.get("constraints")

        if constraints is not None:
            minimum = constraints.get("minimum")
            maximum = constraints.get("maximum")

            if minimum is not None or maximum is not None:
                if minimum is not None and not is_number(minimum):
                    raise InvalidConstraintError(
                        f"Minimum for column '{name}' must be a valid number"
                    )

                if maximum is not None and not is_number(maximum):
                    raise InvalidConstraintError(
                        f"Maximum for column '{name}' must be a valid number"
                    )

                if minimum is not None and maximum is not None and minimum > maximum:
                    raise InvalidConstraintError(
                        f"Minimum for column '{name}' cannot be greater than maximum: "
                        f"{minimum} > {maximum}"
                    )

                if minimum is not None and maximum is not None:
                    values = (minimum, maximum)
                elif minimum is not None:
                    values = (minimum, None)
                else:
                    values = (None, maximum)

                range_columns.append(name)
                range_values.append(values)

    if range_columns:
        failed_count = df.agg(
            *[
                F.sum(
                    F.when(
                        _build_invalid_condition(column, minimum, maximum), 1
                    ).otherwise(0)
                ).alias(column)
                for column, (minimum, maximum) in zip(range_columns, range_values)
            ]
        ).first()

        failed_dict = {
            column: int(failed_count[column] or 0) for column in range_columns
        }
    else:
        failed_dict = {}

    results = []
    if not failed_dict:
        results = []
    else:
        for (key, value), (minimum, maximum) in zip(failed_dict.items(), range_values):
            invalid_conditions = _build_invalid_condition(key, minimum, maximum)
            invalid_rows = (
                df.filter(invalid_conditions).select(key).limit(30)
                if value > 0
                else None
            )

            message = _build_range_message(key, minimum, maximum)

            results.append(
                ValidationResult(
                    rule_name=f"{key}.range",
                    category="range",
                    passed=value == 0,
                    failed_count=value,
                    invalid_rows=invalid_rows,
                    message=message,
                    metadata={
                        "column_name": key,
                        "minimum": minimum,
                        "maximum": maximum,
                    },
                )
            )

    return results
