from collections import Counter
from functools import reduce

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from instacart_etl.validation.models import ValidationResult


def validate_uniqueness(df: DataFrame, *, columns: list[str]) -> ValidationResult:
    if not columns:
        raise ValueError("column names cannot be empty")

    duplicate_keys = [key for key, value in Counter(columns).items() if value > 1]
    if duplicate_keys:
        raise ValueError(
            "input column keys must be unique but got duplicate keys: "
            f"{', '.join(duplicate_keys)}"
        )

    non_null_condition = reduce(
        lambda x, y: x & y, [F.col(column).isNotNull() for column in columns]
    )
    duplicated_values = (
        df.filter(non_null_condition)
        .groupby(columns)
        .count()
        .filter(F.col("count") > 1)
    )
    duplicate_values_count = duplicated_values.count()

    if duplicate_values_count == 0:
        return ValidationResult(
            rule_name=f"{', '.join(columns)}.uniqueness",
            category="uniqueness",
            passed=True,
            message=(f"Columns '{', '.join(columns)}' have no duplicate values"),
            failed_count=0,
            invalid_rows=None,
            metadata={
                "columns": columns,
                "duplicate_values_count": 0,
                "duplicate_rows_count": 0,
            },
        )
    else:
        df = df.filter(non_null_condition)
        duplicate_rows = df.join(
            duplicated_values.select(*columns), on=columns, how="inner"
        )
        duplicate_rows_count = duplicate_rows.count()
        invalid_rows = duplicate_rows.limit(30)

        return ValidationResult(
            rule_name=f"{', '.join(columns)}.uniqueness",
            category="uniqueness",
            passed=False,
            message=(
                f"Columns '{', '.join(columns)}' "
                f"found {duplicate_values_count} duplicate key value(s)"
            ),
            failed_count=duplicate_rows_count,
            invalid_rows=invalid_rows,
            metadata={
                "columns": columns,
                "duplicate_values_count": duplicate_values_count,
                "duplicate_rows_count": duplicate_rows_count,
            },
        )
