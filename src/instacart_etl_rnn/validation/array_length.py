from typing import Any

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

from instacart_etl_rnn.validation.exceptions import InvalidConstraintError
from instacart_etl_rnn.validation.models import ValidationResult
from instacart_etl_rnn.validation.utils import is_positive_integer


def _build_invalid_conditions(
    column_name: str,
    *,
    minimum_length: int | None,
    maximum_length: int | None,
) -> Column:
    invalid_condition = F.lit(False)

    array_size = F.size(F.col(column_name))

    if minimum_length is not None:
        invalid_condition = invalid_condition | (array_size < minimum_length)

    if maximum_length is not None:
        invalid_condition = invalid_condition | (array_size > maximum_length)

    return invalid_condition & F.col(column_name).isNotNull()


def validate_array_lengths(
    df: DataFrame, *, contract: dict[str, Any]
) -> list[ValidationResult]:
    schemas = contract.get("schema")

    normalized_constraint = {}
    for schema in schemas:
        name = schema["name"]
        constraint = schema.get("constraints")
        datatype = schema.get("type")

        if constraint is not None:
            min_length = constraint.get("min_length")
            max_length = constraint.get("max_length")

            if min_length is not None or max_length is not None:
                if "array" not in datatype:
                    raise InvalidConstraintError(
                        f"Column: {name} has constraint length for an array "
                        f"but its data type is {datatype}"
                    )

                if min_length is not None and not is_positive_integer(min_length):
                    raise InvalidConstraintError(
                        "Minimum array length must be a positive integer"
                    )

                if max_length is not None and not is_positive_integer(max_length):
                    raise InvalidConstraintError(
                        "Maximum array length must be a positive integer"
                    )

                if (
                    min_length is not None
                    and max_length is not None
                    and min_length > max_length
                ):
                    raise InvalidConstraintError(
                        f"min_length cannot be greater than max_length "
                        f"for column '{name}': "
                        f"{min_length} > {max_length}"
                    )

                normalized_constraint[name] = (min_length, max_length)

    results = []
    if normalized_constraint:
        failed_count = df.agg(
            *[
                F.sum(
                    F.when(
                        _build_invalid_conditions(
                            key, minimum_length=min_len, maximum_length=max_len
                        ),
                        1,
                    ).otherwise(0)
                ).alias(key)
                for key, (min_len, max_len) in normalized_constraint.items()
            ]
        ).first()

        failed_dict = {
            key: {
                "failed_count": int(failed_count[key] or 0),
                "min_len": value[0],
                "max_len": value[1],
            }
            for key, value in normalized_constraint.items()
        }

        for key, value in failed_dict.items():
            failed_count = value["failed_count"]
            min_len, max_len = value["min_len"], value["max_len"]
            if failed_count > 0:
                invalid_conditions = _build_invalid_conditions(
                    key, minimum_length=min_len, maximum_length=max_len
                )
                invalid_rows = (
                    df.filter(invalid_conditions)
                    .withColumn("actual_length", F.size(F.col(key)))
                    .limit(30)
                )
                message = (
                    f"Column '{key}' contains "
                    f"{failed_count} row(s) that violate the "
                    "configured array length constraint"
                )
            else:
                invalid_rows = None
                message = (
                    f"All non-null arrays in column '{key}' "
                    "satisfy the configured length constraint"
                )

            results.append(
                ValidationResult(
                    rule_name=f"{key}.array_length",
                    category="array_length",
                    passed=failed_count == 0,
                    failed_count=failed_count,
                    invalid_rows=invalid_rows,
                    message=message,
                    metadata={
                        "column_name": key,
                        "min_length": min_len,
                        "max_length": max_len,
                    },
                )
            )

    return results
