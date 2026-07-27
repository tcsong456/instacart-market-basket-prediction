import pytest
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    ByteType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from instacart_etl_rnn.validation.schema import (
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
        (ArrayType(IntegerType()), "array<integer>", True),
        (ArrayType(BooleanType()), "array<bool>", True),
        (ArrayType(StringType()), "array<integer>", False),
        (BooleanType(), "array<bool>", False),
        (ArrayType(BooleanType()), "array<boolean>", True),
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


def test_is_type_compatible_array_unknown_type():
    with pytest.raises(ValueError, match="Unsupported contract type"):
        _is_type_compatible(
            ArrayType(IntegerType()),
            "array<uuid>",
        )


def test_is_type_compatible_array_empty_type():
    with pytest.raises(
        ValueError, match="Array contract type must specify an element type"
    ):
        _is_type_compatible(
            ArrayType(IntegerType()),
            "array<>",
        )


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


def test_validate_column_datatype_array_integer_success(spark):
    schema = StructType(
        [
            StructField("product_history", ArrayType(IntegerType()), nullable=True),
            StructField("aisle_history", ArrayType(BooleanType()), nullable=True),
        ]
    )

    df = spark.createDataFrame(
        [
            (
                [1, 2, 3],
                [True, True],
            ),
            (
                [4, 5],
                [],
            ),
            ([], [False, True]),
        ],
        schema=schema,
    )

    contract = {
        "schema": [
            {
                "name": "product_history",
                "type": "array<integer>",
            },
            {
                "name": "aisle_history",
                "type": "array<bool>",
            },
        ]
    }

    result = validate_column_datatype(
        df,
        contract=contract,
    )

    assert result.passed is True
    assert result.failed_count == 0


def test_validate_column_datatype_array_type_mismatch(spark):
    schema = StructType(
        [StructField("product_history", ArrayType(StringType()), nullable=True)]
    )

    df = spark.createDataFrame(
        [
            (["a", "b"],),
        ],
        schema=schema,
    )

    contract = {
        "schema": [
            {
                "name": "product_history",
                "type": "array<integer>",
            }
        ]
    }

    result = validate_column_datatype(
        df,
        contract=contract,
    )

    assert result.passed is False
    assert result.failed_count == 1

    mismatch = result.metadata["mismatches"][0]

    assert mismatch["column_name"] == "product_history"
    assert mismatch["expected_datatype"] == "array<integer>"
    assert mismatch["actual_datatype"] == ArrayType(StringType())


def test_validate_column_datatype_integer_not_array(spark):
    schema = StructType([StructField("product_history", IntegerType(), nullable=True)])

    df = spark.createDataFrame(
        [
            (1,),
            (2,),
        ],
        schema=schema,
    )

    contract = {
        "schema": [
            {
                "name": "product_history",
                "type": "array<integer>",
            }
        ]
    }

    result = validate_column_datatype(
        df,
        contract=contract,
    )

    assert not result.passed
    assert result.failed_count == 1
    assert result.message == (
        "Incompatible column data types: "
        "product_history: expected array<integer>, but got IntegerType()"
    )


def test_validate_column_datatype_nested_array(spark):
    schema = StructType(
        [StructField("matrix", ArrayType(ArrayType(IntegerType())), False)]
    )

    df = spark.createDataFrame(
        [
            ([[1, 2], [3]],),
        ],
        schema=schema,
    )

    contract = {
        "schema": [
            {
                "name": "matrix",
                "type": "array<array<integer>>",
            }
        ]
    }

    result = validate_column_datatype(
        df,
        contract=contract,
    )

    assert result.passed
