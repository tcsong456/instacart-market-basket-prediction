from typing import Any

from pyspark.sql import Column
from pyspark.sql import functions as F

from instacart_etl_rnn.validation.exceptions import InvalidConstraintError
from instacart_etl_rnn.validation.models import ValidationMetric


def _build_invalid_condition(
    column_name: str,
    allowed_values: list[Any],
) -> Column:
    return F.col(column_name).isNotNull() & ~F.col(column_name).isin(allowed_values)


def allowed_values_validator(
    contract: dict[str, Any],
) -> list[ValidationMetric]:
    metrics: list[ValidationMetric] = []

    for column_schema in contract.get("schema", []):
        column_name = column_schema["name"]

        allowed_values = column_schema.get("constraints", {}).get("allowed_values")

        if allowed_values is None:
            continue

        if not isinstance(allowed_values, list):
            raise InvalidConstraintError(
                f"'allowed_values' for column {column_name!r} must be a list"
            )

        if not allowed_values:
            raise InvalidConstraintError(
                f"'allowed_values' for column {column_name!r} must not be empty"
            )

        invalid_condition = _build_invalid_condition(
            column_name,
            allowed_values,
        )

        alias = f"{column_name}_allowed_values"

        expression = F.sum(
            F.when(
                invalid_condition,
                1,
            ).otherwise(0)
        ).alias(alias)

        metrics.append(
            ValidationMetric(
                alias=alias,
                rule_name=f"{column_name}.allowed_values",
                validation_type="allowed_values",
                columns=(column_name,),
                expression=expression,
            )
        )

    return metrics
