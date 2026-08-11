from typing import Any

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from instacart_etl_rnn.validation.exceptions import InvalidContractError
from instacart_etl_rnn.validation.models import ValidationResult
from instacart_etl_rnn.validation.utils import (
    is_non_empty_string,
    is_non_empty_string_list,
)

SUPPORTED_AGGREGATIONS = {
    "min",
    "max",
    "sum",
    "count",
    "count_distinct",
    "conditional_count",
}


def _group_aggregate_fields(
    aggregate_fields: list[dict[str, Any]],
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped: dict[
        tuple[str, ...],
        list[dict[str, Any]],
    ] = {}

    for field in aggregate_fields:
        key = tuple(field["partition_by"])
        grouped.setdefault(key, []).append(field)

    return grouped


def _build_aggregate_expression(
    field: dict[str, Any],
) -> Column:
    aggregation = field["aggregation"]
    column = field.get("column")

    if aggregation == "min":
        return F.min(column)

    if aggregation == "max":
        return F.max(column)

    if aggregation == "sum":
        return F.sum(column)

    if aggregation == "count":
        return F.count(column)

    if aggregation == "count_distinct":
        return F.countDistinct(column)

    if aggregation == "conditional_count":
        condition = F.coalesce(
            F.expr(field["condition"]),
            F.lit(False),
        )

        return F.sum(F.when(condition, 1).otherwise(0))

    raise InvalidContractError(f"Unsupported aggregation: {aggregation!r}")


def apply_derived_fields(
    df: DataFrame,
    *,
    contract: dict[str, Any],
) -> DataFrame:
    derived_fields = contract.get("derived_fields", [])
    if not isinstance(derived_fields, list):
        raise InvalidContractError("'derived_fields' must be a list")

    if not derived_fields:
        return df

    aggregate_fields = []
    expression_fields = []

    seen_names: set[str] = set()

    for field in derived_fields:
        name = field.get("name")

        if not is_non_empty_string(name):
            raise InvalidContractError("Every derived field must have a non-empty name")

        if name in seen_names:
            raise InvalidContractError(f"Duplicate derived field name: {name!r}")

        seen_names.add(name)

        aggregation = field.get("aggregation")
        expression = field.get("expression")

        if aggregation is None and expression is None:
            raise InvalidContractError(
                f"Derived field {name!r} must define either "
                "'aggregation' or 'expression'"
            )

        if aggregation is not None and expression is not None:
            raise InvalidContractError(
                f"Derived field {name!r} cannot define both "
                "'aggregation' and 'expression'"
            )

        if expression is not None:
            if not is_non_empty_string(expression):
                raise InvalidContractError(
                    f"Expression for derived field {name!r} must be a non-empty string"
                )

            expression_fields.append(field)
            continue

        if aggregation not in SUPPORTED_AGGREGATIONS:
            raise InvalidContractError(
                f"Aggregation {aggregation!r} "
                f"for derived field {name!r} is not supported"
            )

        partition_by = field.get("partition_by")

        if not is_non_empty_string_list(partition_by):
            raise InvalidContractError(
                f"Derived aggregate field {name!r} "
                "must define a non-empty 'partition_by' list"
            )

        if len(partition_by) != len(set(partition_by)):
            raise InvalidContractError(
                f"'partition_by' for derived field {name!r} "
                "cannot contain duplicate columns"
            )

        if aggregation == "conditional_count":
            condition = field.get("condition")

            if not is_non_empty_string(condition):
                raise InvalidContractError(
                    f"Derived field {name!r} using "
                    "'conditional_count' must define "
                    "a non-empty condition"
                )
        else:
            column = field.get("column")

            if not is_non_empty_string(column):
                raise InvalidContractError(
                    f"Derived field {name!r} using {aggregation!r} must define a column"
                )

        aggregate_fields.append(field)

    grouped_fields = _group_aggregate_fields(aggregate_fields)

    for partition_by, fields in grouped_fields.items():
        aggregate_df = df.groupBy(*partition_by).agg(
            *[
                _build_aggregate_expression(field).alias(field["name"])
                for field in fields
            ]
        )

        df = df.join(
            aggregate_df,
            on=list(partition_by),
            how="left",
        )

    for field in expression_fields:
        df = df.withColumn(
            field["name"],
            F.expr(field["expression"]),
        )

    return df


def validate_row_logic(
    df: DataFrame,
    *,
    contract: dict[str, Any],
) -> list[ValidationResult]:
    rules = contract.get("rules", [])
    if not isinstance(rules, list):
        raise InvalidContractError("'rules' must be a list")

    if not rules:
        return []

    derived_df = apply_derived_fields(
        df,
        contract=contract,
    )

    invalid_conditions: dict[str, Column] = {}
    seen_rule_names: set[str] = set()

    for rule in rules:
        name = rule.get("name")

        if not is_non_empty_string(name):
            raise InvalidContractError("Every business rule must have a non-empty name")

        if name in seen_rule_names:
            raise InvalidContractError(f"Duplicate business rule name: {name!r}")

        seen_rule_names.add(name)

        expression = rule.get("expression")

        if not is_non_empty_string(expression):
            raise InvalidContractError(
                f"Rule {name!r} must define a non-empty expression"
            )

        rule_result = F.expr(expression)

        invalid_conditions[name] = rule_result.isNotNull() & ~rule_result

    failed_counts = derived_df.agg(
        *[
            F.sum(F.when(condition, 1).otherwise(0)).alias(name)
            for name, condition in invalid_conditions.items()
        ]
    ).first()

    results = []

    for name, condition in invalid_conditions.items():
        failed_count = int(failed_counts[name] or 0)

        passed = failed_count == 0

        invalid_rows = None if passed else derived_df.filter(condition).limit(20)

        results.append(
            ValidationResult(
                rule_name=name,
                category="row_logic",
                passed=passed,
                failed_count=failed_count,
                invalid_rows=invalid_rows,
                message=(
                    f"Rule {name!r} passed"
                    if passed
                    else (f"Rule {name!r} failed for {failed_count} row(s)")
                ),
                metadata={},
            )
        )

    return results
