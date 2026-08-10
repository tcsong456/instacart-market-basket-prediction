from typing import Any

from pyspark.sql import Column
from pyspark.sql import functions as F

from instacart_etl_rnn.validation.models import ValidationMetric


def nullability_validator(
    contract: dict[str, Any],
) -> list[ValidationMetric]:
    metrics: list[ValidationMetric] = []

    for column_schema in contract.get("schema", []):
        column_name = column_schema["name"]

        nullable = column_schema.get("nullable", True)

        if not isinstance(nullable, bool):
            raise ValueError(f"'nullable' for column {column_name!r} must be a boolean")

        null_threshold = column_schema.get("thresholds", {}).get("nullability")

        if nullable and null_threshold is None:
            continue

        alias = f"{column_name}_nullability"

        expression: Column = F.sum(
            F.when(
                F.col(column_name).isNull(),
                1,
            ).otherwise(0)
        ).alias(alias)

        metrics.append(
            ValidationMetric(
                alias=alias,
                rule_name=f"{column_name}.nullability",
                validation_type="nullability",
                columns=(column_name,),
                expression=expression,
            )
        )

    return metrics
