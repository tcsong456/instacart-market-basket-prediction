import pytest
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from instacart_etl_rnn.bronze.create_bronze_dataset import build_spark_schema


def test_build_spark_schema_returns_expected_schema():
    contract = {
        "schema": [
            {
                "name": "order_id",
                "type": "integer",
                "nullable": False,
            },
            {
                "name": "user_id",
                "type": "long",
                "nullable": False,
            },
            {
                "name": "eval_set",
                "type": "string",
                "nullable": False,
            },
            {
                "name": "days_since_prior_order",
                "type": "double",
                "nullable": True,
            },
            {
                "name": "is_last_order",
                "type": "boolean",
                "nullable": True,
            },
        ]
    }

    expected_schema = StructType(
        [
            StructField(
                "order_id",
                IntegerType(),
                nullable=False,
            ),
            StructField(
                "user_id",
                LongType(),
                nullable=False,
            ),
            StructField(
                "eval_set",
                StringType(),
                nullable=False,
            ),
            StructField(
                "days_since_prior_order",
                DoubleType(),
                nullable=True,
            ),
            StructField(
                "is_last_order",
                BooleanType(),
                nullable=True,
            ),
        ]
    )

    actual_schema = build_spark_schema(contract)

    assert actual_schema == expected_schema


def test_build_spark_schema_rejects_unsupported_type():
    contract = {
        "schema": [
            {
                "name": "created_at",
                "type": "unsupported_type",
                "nullable": False,
            }
        ]
    }

    with pytest.raises(
        ValueError,
        match=("Unsupported contract type 'unsupported_type' for column 'created_at'"),
    ):
        build_spark_schema(contract)
