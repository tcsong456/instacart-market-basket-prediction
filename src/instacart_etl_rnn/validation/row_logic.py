from typing import Any

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from instacart_etl_rnn.validation.exceptions import InvalidContractError
from instacart_etl_rnn.validation.models import ValidationResult


def _group_aggregate_fields(
    aggregate_fields: list[dict[str, Any]],
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    grouped = {}

    for field in aggregate_fields:
        key = tuple(field["partition_by"])

        grouped.setdefault(key, []).append(field)

    return grouped


def _build_aggregate_expression(field: dict[str, Any]) -> Column:
    aggregation = field["aggregation"]
    column = field.get("column")

    if aggregation == "min":
        return F.min(column)
    elif aggregation == "max":
        return F.max(column)
    elif aggregation == "sum":
        return F.sum(column)
    elif aggregation == "conditional_count":
        condition = F.coalesce(F.expr(field["condition"]), F.lit(False))
        return F.sum(F.when(condition, 1).otherwise(0))
    elif aggregation == "count_distinct":
        return F.countDistinct(column)
    elif aggregation == "count":
        return F.count(column)

    raise ValueError(f"Unsupported aggregation: {aggregation}")


def apply_derived_fields(
    df: DataFrame, *, contract: dict[str, Any]
) -> DataFrame | list:
    derived_fields = contract.get("derived_fields", [])
    if not derived_fields:
        return df

    aggregate_fields, expression_fields = [], []
    for field in derived_fields:
        if "aggregation" not in field and "expression" not in field:
            raise InvalidContractError(
                "Missing both aggregation and expression. "
                "In derived field, there must be either one of them as key"
            )

        agg = field.get("aggregation")
        if agg and agg not in [
            "min",
            "max",
            "count",
            "conditional_count",
            "sum",
            "count_distinct",
        ]:
            raise InvalidContractError(f"Aggregation type {agg} is not supported")

        if agg and agg != "conditional_count" and "column" not in field:
            raise InvalidContractError(
                f"Derived column: {field['name']} does not provide its "
                "aggregated column"
            )

        exp = field.get("expression")
        if exp and agg:
            raise InvalidContractError(
                "Only one of aggregation or expression should be given "
                "to the derived column rule"
            )

        if exp is not None:
            expression_fields.append(field)

        if agg is not None:
            aggregate_fields.append(field)

    grouped_fields = _group_aggregate_fields(aggregate_fields)

    for group, group_field in grouped_fields.items():
        agg_df = df.groupby(*group).agg(
            *[
                _build_aggregate_expression(field).alias(field["name"])
                for field in group_field
            ]
        )

        df = df.join(agg_df, how="left", on=list(group))

    for field in expression_fields:
        df = df.withColumn(field["name"], F.expr(field["expression"]))

    return df


def validate_row_logic(
    df: DataFrame, contract: dict[str, Any]
) -> list[ValidationResult]:
    results = []
    rules = contract.get("rules")
    if not rules:
        return results

    derived_df = apply_derived_fields(df, contract=contract)

    invalid_conditions = {}
    for rule in rules:
        if "name" not in rule:
            raise InvalidContractError("Every rule must have a name")

        if "expression" not in rule:
            raise InvalidContractError(
                f"Rule: {rule['name']} does not have its expression"
            )
        
        rule_result = F.expr(rule["expression"])
        invalid_conditions = rule_result.isNotNull() & ~rule_result

    failed_counts = derived_df.agg(
        *[
            F.sum(F.when(rule_condition, 1).otherwise(0)).alias(rule_name)
            for rule_name, rule_condition in invalid_conditions.items()
        ]
    ).first()

    for name, condition in invalid_conditions.items():
        failed_count = int(failed_counts[name] or 0)

        if failed_count > 0:
            invalid_rows = derived_df.filter(condition).limit(20)
        else:
            invalid_rows = None

        passed = failed_count == 0
        message = (
            f"Rule: {name} complies with the data contract rule"
            if passed
            else f"Rule: {name} failed the data contract rule"
        )

        results.append(
            ValidationResult(
                rule_name=name,
                category="row_logic",
                passed=passed,
                invalid_rows=invalid_rows,
                failed_count=failed_count,
                message=message,
                metadata={},
            )
        )

    return results
