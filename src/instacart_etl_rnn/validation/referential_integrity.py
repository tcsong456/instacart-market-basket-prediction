from functools import reduce
from operator import or_
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from instacart_etl_rnn.validation.exceptions import InvalidContractError
from instacart_etl_rnn.validation.models import ValidationResult


def _validate_referential_integrity_arguments(
    *,
    child_df: DataFrame,
    parent_df: DataFrame,
    child_columns: list[str],
    parent_columns: list[str],
) -> None:
    if len(child_columns) != len(set(child_columns)):
        raise InvalidContractError(
            "child_columns cannot contain duplicate column names"
        )

    if len(parent_columns) != len(set(parent_columns)):
        raise InvalidContractError(
            "parent_columns cannot contain duplicate column names"
        )

    missing_child_columns = [
        column for column in child_columns if column not in child_df.columns
    ]

    if missing_child_columns:
        raise InvalidContractError(
            f"Child DataFrame is missing column(s): {missing_child_columns}"
        )

    missing_parent_columns = [
        column for column in parent_columns if column not in parent_df.columns
    ]

    if missing_parent_columns:
        raise InvalidContractError(
            f"Parent DataFrame is missing column(s): {missing_parent_columns}"
        )


def extract_foreign_keys(
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    relationships = contract.get("relationships", [])

    foreign_keys = []

    for relationship in relationships:
        if relationship.get("type") != "foreign_key":
            continue

        child_columns = relationship.get("child_columns")
        parent = relationship.get("parent")

        if (
            not isinstance(child_columns, list)
            or not child_columns
            or not all(
                isinstance(column, str) and column.strip() for column in child_columns
            )
        ):
            raise InvalidContractError(
                "Foreign-key relationship must contain a non-empty 'child_columns' list"
            )

        if not isinstance(parent, dict):
            raise InvalidContractError(
                "Foreign-key relationship must contain a 'parent' mapping"
            )

        parent_dataset = parent.get("dataset")
        parent_columns = parent.get("columns")

        if not isinstance(parent_dataset, str) or not parent_dataset.strip():
            raise InvalidContractError(
                "Foreign-key parent must contain a valid dataset name"
            )

        if (
            not isinstance(parent_columns, list)
            or not parent_columns
            or not all(
                isinstance(column, str) and column.strip() for column in parent_columns
            )
        ):
            raise InvalidContractError(
                "Foreign-key parent must contain a non-empty 'columns' list"
            )

        if len(child_columns) != len(parent_columns):
            raise InvalidContractError(
                "Foreign-key child_columns and parent columns must have the same length"
            )

        foreign_keys.append(
            {
                "name": relationship.get(
                    "name",
                    f"{'_'.join(child_columns)}_fk",
                ),
                "child_columns": child_columns,
                "parent_dataset": parent_dataset,
                "parent_columns": parent_columns,
            }
        )

    return foreign_keys


def _validate_referential_integrity(
    *,
    child_df: DataFrame,
    parent_df: DataFrame,
    child_columns: list[str],
    parent_columns: list[str],
    rule_name: str = "",
) -> ValidationResult:
    _validate_referential_integrity_arguments(
        child_df=child_df,
        parent_df=parent_df,
        child_columns=child_columns,
        parent_columns=parent_columns,
    )

    child_nulls = reduce(or_, (F.col(column).isNull() for column in child_columns))
    non_null_child = child_df.filter(~child_nulls)

    parent_keys = parent_df.select(
        *[
            F.col(parent_column).alias(child_column)
            for child_column, parent_column in zip(
                child_columns, parent_columns, strict=True
            )
        ]
    ).distinct()

    invalid_rows = non_null_child.join(parent_keys, on=child_columns, how="left_anti")

    failed_count = invalid_rows.count()
    passed = failed_count == 0

    if not passed:
        invalid_rows = invalid_rows.select(*child_columns).distinct().limit(20)
    else:
        invalid_rows = None

    child_key = ", ".join(child_columns)
    parent_key = ", ".join(parent_columns)

    return ValidationResult(
        rule_name=rule_name,
        category="referential_integrity",
        passed=passed,
        message=(
            f"All non-null child key values '{child_key}' exist "
            f"in parent columns '{parent_key}'"
            if passed
            else (
                f"Found {failed_count} child row(s) whose key values "
                f"'{child_key}' do not exist in parent columns "
                f"'{parent_key}'"
            )
        ),
        failed_count=failed_count,
        invalid_rows=invalid_rows,
        metadata={"child_columns": child_columns, "parent_columns": parent_columns},
    )


def validate_foreign_keys(
    *,
    child_df: DataFrame,
    contract: dict[str, Any],
    datasets: dict[str, DataFrame],
) -> list[ValidationResult]:
    results = []

    for foreign_key in extract_foreign_keys(contract):
        parent_dataset = foreign_key["parent_dataset"]

        if parent_dataset not in datasets:
            raise ValueError(f"Parent dataset {parent_dataset!r} is not available")

        result = _validate_referential_integrity(
            child_df=child_df,
            parent_df=datasets[parent_dataset],
            child_columns=foreign_key["child_columns"],
            parent_columns=foreign_key["parent_columns"],
            rule_name=foreign_key["name"],
        )

        results.append(result)

    return results
