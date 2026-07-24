from typing import Any

from pyspark.sql import DataFrame

from instacart_etl.validation.models import ValidationResult


def validate_columns(
    df: DataFrame,
    *,
    contract: dict[str, Any],
) -> ValidationResult:
    expected_columns = set(s["name"] for s in contract["schema"])
    actual_columns = set(df.columns)

    allow_extra_columns = contract.get("dataset", {}).get("allow_extra_columns", False)

    missing_columns = sorted(expected_columns - actual_columns)
    unexpected_columns = sorted(actual_columns - expected_columns)

    passed = not missing_columns and (allow_extra_columns or not unexpected_columns)

    failed_count = len(missing_columns) + (
        0 if allow_extra_columns else len(unexpected_columns)
    )

    unexpected_columns_str = ", ".join(unexpected_columns) or "none"
    if allow_extra_columns and unexpected_columns:
        unexpected_message = f"{unexpected_columns_str} (ignored)"
    else:
        unexpected_message = unexpected_columns_str

    message = (
        "all required columns are present"
        if passed
        else f"missing columns: {', '.join(missing_columns) or 'none'}; "
        f"unexpected columns: {unexpected_message}"
    )

    return ValidationResult(
        rule_name="column_presence",
        category="schema",
        passed=passed,
        message=message,
        failed_count=failed_count,
        invalid_rows=None,
        metadata={
            "expected_columns": sorted(expected_columns),
            "unexpected_columns": unexpected_columns,
            "missing_columns": missing_columns,
            "actual_columns": sorted(df.columns),
        },
    )
