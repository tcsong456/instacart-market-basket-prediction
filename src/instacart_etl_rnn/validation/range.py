from typing import Any

from pyspark.sql import Column
from pyspark.sql import functions as F

from instacart_etl_rnn.validation.exceptions import InvalidConstraintError
from instacart_etl_rnn.validation.models import ValidationMetric
from instacart_etl_rnn.validation.utils import is_number


def _build_range_invalid_condition(
    column_name: str,
    *,
    minimum: int | float | None,
    maximum: int | float | None,
) -> Column:
    condition = F.lit(False)

    if minimum is not None:
        condition |= F.col(column_name) < F.lit(minimum)

    if maximum is not None:
        condition |= F.col(column_name) > F.lit(maximum)

    return F.col(column_name).isNotNull() & condition


def range_validator(
    contract: dict[str, Any],
) -> list[ValidationMetric]:
    metrics: list[ValidationMetric] = []

    for column_schema in contract.get("schema", []):
        column_name = column_schema["name"]
        constraints = column_schema.get("constraints", {})

        minimum = constraints.get("minimum")
        maximum = constraints.get("maximum")

        if minimum is None and maximum is None:
            continue

        if minimum is not None and not is_number(minimum):
            raise InvalidConstraintError(
                f"minimum for column {column_name!r} must be a valid number"
            )

        if maximum is not None and not is_number(maximum):
            raise InvalidConstraintError(
                f"maximum for column {column_name!r} must be a valid number"
            )

        if minimum is not None and maximum is not None and minimum > maximum:
            raise InvalidConstraintError(
                f"minimum cannot be greater than maximum "
                f"for column {column_name!r}: "
                f"{minimum} > {maximum}"
            )

        alias = f"{column_name}_range"

        invalid_condition = _build_range_invalid_condition(
            column_name,
            minimum=minimum,
            maximum=maximum,
        )

        expression = F.sum(
            F.when(
                invalid_condition,
                1,
            ).otherwise(0)
        ).alias(alias)

        metrics.append(
            ValidationMetric(
                alias=alias,
                rule_name=f"{column_name}.range",
                validation_type="range",
                columns=(column_name,),
                expression=expression,
            )
        )

    return metrics
