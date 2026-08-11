import pytest
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    ByteType,
    DateType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
    StringType,
    TimestampType,
)

from instacart_etl_rnn.validation.exceptions import InvalidContractError
from instacart_etl_rnn.validation.schema import _is_type_compatible


@pytest.mark.parametrize(
    "actual_type,contract_type",
    [
        (ByteType(), "integer"),
        (ShortType(), "integer"),
        (IntegerType(), "integer"),
        (LongType(), "integer"),
        (IntegerType(), "int"),
        (FloatType(), "float"),
        (FloatType(), "double"),
        (DoubleType(), "double"),
        (IntegerType(), "number"),
        (LongType(), "number"),
        (FloatType(), "number"),
        (DoubleType(), "number"),
        (DecimalType(10, 2), "number"),
        (StringType(), "string"),
        (StringType(), "str"),
        (BooleanType(), "boolean"),
        (BooleanType(), "bool"),
        (DateType(), "date"),
        (TimestampType(), "timestamp"),
        (DecimalType(10, 2), "decimal"),
    ],
)
def test_is_type_compatible_accepts_compatible_types(
    actual_type,
    contract_type,
):
    assert _is_type_compatible(
        actual_type,
        contract_type,
    )


@pytest.mark.parametrize(
    "actual_type,contract_type",
    [
        (StringType(), "integer"),
        (DoubleType(), "integer"),
        (IntegerType(), "string"),
        (StringType(), "boolean"),
        (DateType(), "timestamp"),
        (DoubleType(), "float"),
    ],
)
def test_is_type_compatible_rejects_incompatible_types(
    actual_type,
    contract_type,
):
    assert not _is_type_compatible(
        actual_type,
        contract_type,
    )


@pytest.mark.parametrize(
    "contract_type",
    [
        "INTEGER",
        "Integer",
        " integer ",
        "InTeGeR",
    ],
)
def test_is_type_compatible_normalizes_contract_type(contract_type):
    assert _is_type_compatible(
        IntegerType(),
        contract_type,
    )


def test_is_type_compatible_accepts_array_type():
    assert _is_type_compatible(
        ArrayType(IntegerType()),
        "array<integer>",
    )


def test_is_type_compatible_accepts_nested_array_type():
    assert _is_type_compatible(
        ArrayType(ArrayType(StringType())),
        "array<array<string>>",
    )


def test_is_type_compatible_rejects_wrong_array_element_type():
    assert not _is_type_compatible(
        ArrayType(StringType()),
        "array<integer>",
    )


def test_is_type_compatible_rejects_non_array_for_array_contract():
    assert not _is_type_compatible(
        IntegerType(),
        "array<integer>",
    )


def test_is_type_compatible_rejects_array_without_element_type():
    with pytest.raises(
        InvalidContractError,
        match="must specify an element type",
    ):
        _is_type_compatible(
            ArrayType(IntegerType()),
            "array<>",
        )


def test_is_type_compatible_rejects_unsupported_contract_type():
    with pytest.raises(
        InvalidContractError,
        match="Unsupported contract type",
    ):
        _is_type_compatible(
            StringType(),
            "whatever",
        )
