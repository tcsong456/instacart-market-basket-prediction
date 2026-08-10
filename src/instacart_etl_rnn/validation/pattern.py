from typing import Any

from pyspark.sql import Column
from pyspark.sql import functions as F

from instacart_etl_rnn.validation.exceptions import InvalidConstraintError
from instacart_etl_rnn.validation.models import ValidationMetric


def _build_invalid_condition(
    column_name: str,
    pattern: str,
) -> Column:
    return F.col(column_name).isNotNull() & ~F.col(column_name).rlike(pattern)


def pattern_validator(
    contract: dict[str, Any],
) -> list[ValidationMetric]:
    metrics: list[ValidationMetric] = []

    for column_schema in contract.get("schema", []):
        column_name = column_schema["name"]

        pattern = column_schema.get("constraints", {}).get("pattern")

        if pattern is None:
            continue

        if not isinstance(pattern, str):
            raise InvalidConstraintError(
                f"pattern for column {column_name!r} must be a string"
            )

        if not pattern.strip():
            raise InvalidConstraintError(
                f"pattern for column {column_name!r} "
                "must not be empty or whitespace-only"
            )

        invalid_condition = _build_invalid_condition(
            column_name,
            pattern,
        )

        alias = f"{column_name}_pattern"

        expression = F.sum(
            F.when(
                invalid_condition,
                1,
            ).otherwise(0)
        ).alias(alias)

        metrics.append(
            ValidationMetric(
                alias=alias,
                rule_name=f"{column_name}.pattern",
                validation_type="pattern",
                columns=(column_name,),
                expression=expression,
            )
        )

    return metrics
