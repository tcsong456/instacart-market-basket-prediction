from collections import Counter
from typing import Any

from pyspark.sql import DataFrame
from pyspark.sql.types import (
    ArrayType,
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

from instacart_etl_rnn.validation.exceptions import InvalidContractError
from instacart_etl_rnn.validation.models import ValidationResult


def _find_duplicates(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def validate_column_presence(
    df: DataFrame,
    *,
    contract: dict[str, Any],
) -> ValidationResult:
    expected_column_list = [schema["name"] for schema in contract["schema"]]
    actual_column_list = df.columns

    duplicate_expected = _find_duplicates(expected_column_list)
    duplicate_actual = _find_duplicates(actual_column_list)

    if duplicate_expected:
        raise InvalidContractError(
            f"Duplicate columns in contract schema: {', '.join(duplicate_expected)}"
        )

    expected_columns = set(expected_column_list)
    actual_columns = set(actual_column_list)

    allow_extra_columns = contract.get("dataset", {}).get("allow_extra_columns", False)

    if not isinstance(allow_extra_columns, bool):
        raise InvalidContractError("'allow_extra_columns' must be a boolean")

    missing_columns = sorted(expected_columns - actual_columns)
    unexpected_columns = sorted(actual_columns - expected_columns)

    passed = (
        not missing_columns
        and not duplicate_actual
        and (allow_extra_columns or not unexpected_columns)
    )

    failed_count = (
        len(missing_columns)
        + len(duplicate_actual)
        + (0 if allow_extra_columns else len(unexpected_columns))
    )

    missing_columns_str = ", ".join(missing_columns) or "none"
    unexpected_columns_str = ", ".join(unexpected_columns) or "none"

    if allow_extra_columns and unexpected_columns:
        unexpected_columns_str += " (ignored)"

    message = (
        f"missing columns: {missing_columns_str}; "
        f"unexpected columns: {unexpected_columns_str}"
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
            "duplicate_columns": duplicate_actual,
        },
    )


def _is_type_compatible(
    actual_type: DataType | None,
    contract_type: str,
) -> bool:
    if actual_type is None:
        return False

    normalized_type = contract_type.strip().lower()

    if normalized_type.startswith("array<") and normalized_type.endswith(">"):
        if not isinstance(actual_type, ArrayType):
            return False

        element_contract_type = normalized_type[len("array<") : -1].strip()

        if not element_contract_type:
            raise InvalidContractError(
                f"Array contract type must specify an element type: {contract_type!r}"
            )

        return _is_type_compatible(
            actual_type.elementType,
            element_contract_type,
        )

    compatibility_map = {
        "integer": (
            ByteType,
            ShortType,
            IntegerType,
            LongType,
        ),
        "int": (
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
        "str": (StringType,),
        "boolean": (BooleanType,),
        "bool": (BooleanType,),
        "date": (DateType,),
        "timestamp": (TimestampType,),
        "decimal": (DecimalType,),
    }

    accepted_types = compatibility_map.get(normalized_type)

    if accepted_types is None:
        raise InvalidContractError(f"Unsupported contract type: {contract_type!r}")

    return isinstance(actual_type, accepted_types)


def validate_column_datatype(
    df: DataFrame,
    *,
    contract: dict[str, Any],
) -> ValidationResult:
    actual_schemas = {field.name: field.dataType for field in df.schema.fields}

    mismatches = []

    for contract_schema in contract["schema"]:
        column_name = contract_schema["name"]
        contract_datatype = contract_schema["type"]

        actual_datatype = actual_schemas.get(column_name)

        if actual_datatype is None:
            continue

        if not _is_type_compatible(
            actual_datatype,
            contract_datatype,
        ):
            mismatches.append(
                {
                    "column_name": column_name,
                    "actual_datatype": actual_datatype,
                    "expected_datatype": contract_datatype,
                }
            )

    passed = not mismatches

    if passed:
        message = "All columns have compatible data types"
    else:
        mismatch_text = "; ".join(
            f"{mismatch['column_name']}: expected "
            f"{mismatch['expected_datatype']}, "
            f"but got {mismatch['actual_datatype']}"
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
        },
    )
