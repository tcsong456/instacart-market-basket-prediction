from collections import Counter
from functools import reduce
from operator import and_

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from instacart_etl_rnn.validation.exceptions import InvalidContractError
from instacart_etl_rnn.validation.models import ValidationResult
from instacart_etl_rnn.validation.utils import is_non_empty_string_list


def validate_uniqueness(
    df: DataFrame,
    *,
    columns: list[str],
) -> ValidationResult:
    if not is_non_empty_string_list(columns):
        raise InvalidContractError("columns must be a non-empty list of column names")

    duplicate_columns = [
        column for column, count in Counter(columns).items() if count > 1
    ]

    if duplicate_columns:
        raise InvalidContractError(
            "columns cannot contain duplicate column names: "
            f"{', '.join(duplicate_columns)}"
        )

    non_null_condition = reduce(
        and_,
        (F.col(column).isNotNull() for column in columns),
    )

    non_null_df = df.filter(non_null_condition)

    duplicate_keys = (
        non_null_df.groupBy(*columns)
        .count()
        .filter(F.col("count") > 1)
        .select(*columns)
    )

    duplicate_rows = non_null_df.join(
        duplicate_keys,
        on=columns,
        how="inner",
    )

    duplicate_row_count = duplicate_rows.count()
    duplicate_key_count = duplicate_keys.count()

    rule_name = f"{', '.join(columns)}.uniqueness"
    column_names = ", ".join(columns)

    if duplicate_row_count == 0:
        return ValidationResult(
            rule_name=rule_name,
            category="uniqueness",
            passed=True,
            message=(f"Columns '{column_names}' have no duplicate values"),
            failed_count=0,
            invalid_rows=None,
            metadata={
                "columns": columns,
                "duplicate_row_count": 0,
                "duplicate_key_count": 0,
            },
        )

    return ValidationResult(
        rule_name=rule_name,
        category="uniqueness",
        passed=False,
        message=(
            f"Columns '{column_names}' have "
            f"{duplicate_row_count} row(s) participating "
            "in duplicate key values"
        ),
        failed_count=duplicate_row_count,
        invalid_rows=duplicate_rows.limit(30),
        metadata={
            "columns": columns,
            "duplicate_row_count": duplicate_row_count,
            "duplicate_key_count": duplicate_key_count,
        },
    )
