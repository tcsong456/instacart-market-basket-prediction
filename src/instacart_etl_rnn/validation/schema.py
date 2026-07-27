from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql.types import (
    BooleanType,
    ByteType,
    DataType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    NumericType,
    ShortType,
    StringType,
    TimestampType,
)

from instacart_etl_rnn.validation.models import ValidationResult


def validate_column_presence(
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


def _is_type_compatible(
    actual_type: DataType,
    contract_type: str,
) -> bool:
    normalized_type = contract_type.strip().lower()

    compatibility_map = {
        "integer": (
            ByteType,
            ShortType,
            IntegerType,
            LongType,
        ),
        "float": (FloatType,),
        "double": (
            FloatType,
            DoubleType,
        ),
        "number": (NumericType,),
        "string": (StringType,),
        "boolean": (BooleanType,),
        "date": (DateType,),
        "timestamp": (TimestampType,),
        "decimal": (DecimalType,),
    }

    accepted_types = compatibility_map.get(normalized_type)

    if accepted_types is None:
        raise ValueError(f"Unsupported contract type: {contract_type!r}")

    return isinstance(actual_type, accepted_types)


def validate_column_datatype(
    df: DataFrame, *, contract: dict[str, Any]
) -> ValidationResult:
    actual_schemas = {field.name: field.dataType for field in df.schema.fields}

    mismatches, skipped_columns = [], []
    for contract_schema in contract["schema"]:
        contract_datatype = contract_schema["type"]
        contract_col_name = contract_schema["name"]

        actual_datatype = actual_schemas.get(contract_col_name)
        if actual_datatype is None:
            skipped_columns.append(contract_col_name)
            continue

        if not _is_type_compatible(actual_datatype, contract_datatype):
            mismatches.append(
                {
                    "column_name": contract_col_name,
                    "actual_datatype": actual_datatype,
                    "expected_datatype": contract_datatype,
                }
            )

    passed = len(mismatches) == 0

    if passed:
        message = "All columns have compatible data types"
    else:
        mismatch_text = "; ".join(
            (
                f"{mismatch['column_name']}: expected "
                f"{mismatch['expected_datatype']}, but got "
                f"{mismatch['actual_datatype']}"
            )
            for mismatch in mismatches
        )
        message = f"Incompatible column data types: {mismatch_text}"

    return ValidationResult(
        rule_name="column_types",
        category="schema",
        passed=passed,
        message=message,
        failed_count=len(mismatches),
        invalid_rows=None,
        metadata={
            "mismatches": mismatches,
            "skipped_missing_columns": skipped_columns,
        },
    )
