from typing import Any

from pyspark.sql import Column
from pyspark.sql import functions as F

from instacart_etl_rnn.validation.exceptions import InvalidConstraintError
from instacart_etl_rnn.validation.models import ValidationMetric
from instacart_etl_rnn.validation.utils import is_non_negative_integer


def _build_invalid_condition(
    column_name: str,
    *,
    minimum: int | None,
    maximum: int | None,
) -> Column:
    invalid_condition = F.lit(False)

    if minimum is not None:
        invalid_condition = invalid_condition | (
            F.size(F.col(column_name)) < F.lit(minimum)
        )

    if maximum is not None:
        invalid_condition = invalid_condition | (
            F.size(F.col(column_name)) > F.lit(maximum)
        )

    return F.col(column_name).isNotNull() & invalid_condition


def array_length_validator(
    contract: dict[str, Any],
) -> list[ValidationMetric]:
    metrics: list[ValidationMetric] = []

    for column_schema in contract.get("schema", []):
        column_name = column_schema["name"]

        constraints = column_schema.get(
            "constraints",
            {},
        )

        minimum = constraints.get("minimum_array_length")
        maximum = constraints.get("maximum_array_length")

        if minimum is None and maximum is None:
            continue

        if minimum is not None and not is_non_negative_integer(minimum):
            raise InvalidConstraintError(
                f"minimum array length for column "
                f"{column_name!r} must be a "
                "non-negative integer"
            )

        if maximum is not None and not is_non_negative_integer(maximum):
            raise InvalidConstraintError(
                f"maximum array length for column "
                f"{column_name!r} must be a "
                "non-negative integer"
            )

        if minimum is not None and maximum is not None and minimum > maximum:
            raise InvalidConstraintError(
                f"minimum array length cannot be greater "
                f"than maximum array length for column "
                f"{column_name!r}"
            )

        invalid_condition = _build_invalid_condition(
            column_name,
            minimum=minimum,
            maximum=maximum,
        )

        alias = f"{column_name}_array_length"

        expression = F.sum(
            F.when(
                invalid_condition,
                1,
            ).otherwise(0)
        ).alias(alias)

        metrics.append(
            ValidationMetric(
                alias=alias,
                rule_name=f"{column_name}.array_length",
                validation_type="array_length",
                columns=(column_name,),
                expression=expression,
            )
        )

    return metrics
