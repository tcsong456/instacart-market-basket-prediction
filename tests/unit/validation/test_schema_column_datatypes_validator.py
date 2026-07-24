import pytest
from pyspark.sql.types import (
    ByteType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from instacart_etl.validation.schema import (
    _is_type_compatible,
    validate_column_datatype,
)


@pytest.mark.parametrize(
    ("actual_type", "contract_type", "expected_pass"),
    [
        (IntegerType(), "integer", True),
        (LongType(), "integer", True),
        (StringType(), "Integer", False),
        (DoubleType(), "douBle", True),
        (StringType(), "STRING", True),
        (FloatType(), "integer", False),
    ],
)
def test_is_type_compatible(
    actual_type,
    contract_type,
    expected_pass,
):
    assert _is_type_compatible(actual_type, contract_type) is expected_pass


def test_is_type_compatible_value_error():
    with pytest.raises(ValueError, match="Unsupported contract type"):
        _is_type_compatible(IntegerType(), "bigint")


@pytest.mark.parametrize(
    ("actual_datatype", "contract_datatype", "passed"),
    [
        (LongType(), "integer", True),
        (ByteType(), "integer", True),
        (IntegerType(), "INTEGER", True),
        (StringType(), "double", False),
        (StringType(), "string", True),
        (DoubleType(), "integer", False),
        (FloatType(), "double", True),
        (IntegerType(), "sTriNg", False),
    ],
)
def test_validate_single_column_datatype(
    spark, actual_datatype, contract_datatype, passed
):
    schema = StructType([StructField("test_column", actual_datatype, nullable=True)])
    df = spark.createDataFrame([], schema=schema)

    contract = {
        "schema": [
            {
                "name": "test_column",
                "type": contract_datatype,
            }
        ]
    }

    result = validate_column_datatype(df, contract=contract)

    assert result.passed is passed
    assert result.failed_count == (0 if passed else 1)


def test_validate_multiple_column_datatype(spark):
    schema = StructType(
        [
            StructField("test_column_1", IntegerType(), nullable=True),
            StructField("test_column_2", StringType(), nullable=True),
            StructField("test_column_3", DoubleType(), nullable=True),
            StructField("test_column_4", FloatType(), nullable=True),
        ]
    )
    df = spark.createDataFrame([], schema=schema)

    contract = {
        "schema": [
            {"name": "test_column_1", "type": "integer"},
            {"name": "test_column_2", "type": "integer"},
            {"name": "test_column_3", "type": "float"},
            {"name": "test_column_4", "type": "double"},
            {"name": "test_column_5", "type": "boolean"},
        ]
    }

    result = validate_column_datatype(df, contract=contract)

    assert result.rule_name == "column_types"
    assert result.category == "schema"
    assert result.passed is False
    assert result.message == (
        "Incompatible column data types: "
        "test_column_2: expected integer, but got StringType(); "
        "test_column_3: expected float, but got DoubleType()"
    )
    assert result.failed_count == 2
    assert result.invalid_rows is None
    assert result.metadata["mismatches"] == [
        {
            "column_name": "test_column_2",
            "actual_datatype": StringType(),
            "expected_datatype": "integer",
        },
        {
            "column_name": "test_column_3",
            "actual_datatype": DoubleType(),
            "expected_datatype": "float",
        },
    ]
    assert result.metadata["skipped_missing_columns"] == ["test_column_5"]
