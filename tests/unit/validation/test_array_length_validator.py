import pytest
from pyspark.sql import Row
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from instacart_etl_rnn.validation.array_length import validate_array_lengths
from instacart_etl_rnn.validation.exceptions import InvalidConstraintError


def test_validate_array_length_all_pass(spark):
    df = spark.createDataFrame(
        [(["3_0", "2_3_4"], [1, 3, -100]), ([""], [2, 4])],
        ["order_history", "product_history"],
    )

    contract = {
        "schema": [
            {
                "name": "order_history",
                "type": "array<str>",
                "constraints": {"min_length": 1},
            },
            {
                "name": "product_history",
                "type": "array<int>",
                "constraints": {"max_length": 3},
            },
        ]
    }

    results = validate_array_lengths(df, contract=contract)

    assert results[0].rule_name == "order_history.array_length"
    assert results[0].category == "array_length"
    assert results[0].passed is True
    assert results[0].failed_count == 0
    assert results[0].invalid_rows is None
    assert results[0].message == (
        "All non-null arrays in column 'order_history' "
        "satisfy the configured length constraint"
    )
    assert results[0].metadata == {
        "column_name": "order_history",
        "min_length": 1,
        "max_length": None,
    }

    assert results[1].rule_name == "product_history.array_length"
    assert results[1].category == "array_length"
    assert results[1].passed is True
    assert results[1].failed_count == 0
    assert results[1].invalid_rows is None
    assert results[1].message == (
        "All non-null arrays in column 'product_history' "
        "satisfy the configured length constraint"
    )
    assert results[1].metadata == {
        "column_name": "product_history",
        "min_length": None,
        "max_length": 3,
    }


def test_validate_array_length_all_fail(spark):
    df = spark.createDataFrame(
        [
            (
                ["3_0", "2_3_4"],
                [1, 3, -100, 50],
            ),
            (
                [],
                [2, 4],
            ),
        ],
        ["order_history", "product_history"],
    )

    contract = {
        "schema": [
            {
                "name": "order_history",
                "type": "array<str>",
                "constraints": {"min_length": 1},
            },
            {
                "name": "product_history",
                "type": "array<int>",
                "constraints": {"max_length": 3},
            },
        ]
    }

    results = validate_array_lengths(df, contract=contract)

    assert results[0].passed is False
    assert results[0].failed_count == 1
    assert results[0].message == (
        "Column 'order_history' contains 1 row(s) that violate the "
        "configured array length constraint"
    )
    invalid_rows = {
        tuple(row["order_history"]) for row in results[0].invalid_rows.collect()
    }
    assert invalid_rows == {()}

    assert results[1].passed is False
    assert results[1].failed_count == 1
    assert results[1].message == (
        "Column 'product_history' contains 1 row(s) that violate the "
        "configured array length constraint"
    )
    invalid_rows = {
        tuple(row["product_history"]) for row in results[1].invalid_rows.collect()
    }
    assert invalid_rows == {(1, 3, -100, 50)}


def test_validate_array_length_null_ignored(spark):
    df = spark.createDataFrame(
        [
            (
                ["3_0"],
                None,
                None,
            ),
            (
                None,
                [1, 2, 3, 4, 5],
                None,
            ),
            (
                ["10_20", "2_3_4"],
                [1, 3, -100, 50],
                None,
            ),
        ],
        ["order_history", "product_history", "aisle_history"],
    )

    contract = {
        "schema": [
            {
                "name": "order_history",
                "type": "array<str>",
                "constraints": {"min_length": 1},
            },
            {
                "name": "product_history",
                "type": "array<int>",
                "constraints": {"max_length": 3},
            },
            {
                "name": "aisle_history",
                "type": "array<int>",
                "constraints": {"min_length": 3, "max_length": 10},
            },
        ]
    }

    results = validate_array_lengths(df, contract=contract)

    assert results[0].passed is True
    assert results[0].failed_count == 0

    assert results[1].passed is False
    assert results[1].failed_count == 2
    invalid_rows = {
        tuple(row["product_history"]) for row in results[1].invalid_rows.collect()
    }
    assert invalid_rows == {(1, 2, 3, 4, 5), (1, 3, -100, 50)}

    assert results[2].passed is True
    assert results[2].failed_count == 0


def test_validate_array_length_with_both_boundaries(spark):
    df = spark.createDataFrame(
        [
            (
                ["3_0", "2_3_4"],
                [1, 3, -100, 50],
            ),
            (
                ["1_2"],
                [2, 4, -1, 100, -50],
            ),
            (
                ["3_0", "2_3_4", ""],
                [1000],
            ),
        ],
        ["order_history", "product_history"],
    )

    contract = {
        "schema": [
            {
                "name": "order_history",
                "type": "array<str>",
                "constraints": {"min_length": 1, "max_length": 3},
            },
            {
                "name": "product_history",
                "type": "array<int>",
                "constraints": {"min_length": 2, "max_length": 4},
            },
        ]
    }

    results = validate_array_lengths(df, contract=contract)

    assert results[0].passed is True

    assert results[1].passed is False
    assert results[1].failed_count == 2
    invalid_rows = {
        tuple(row["product_history"]) for row in results[1].invalid_rows.collect()
    }
    assert invalid_rows == {(1000,), (2, 4, -1, 100, -50)}


def test_validate_array_length_invalid_datatype(spark):
    df = spark.createDataFrame([(["3_0", "2_3_4"],)], ["order_history"])

    contract = {
        "schema": [
            {
                "name": "order_history",
                "type": "integer",
                "constraints": {"min_length": 1, "max_length": 3},
            },
        ]
    }

    with pytest.raises(
        InvalidConstraintError, match="has constraint length for an array"
    ):
        validate_array_lengths(df, contract=contract)


@pytest.mark.parametrize(
    ("raw_data", "min_length"),
    [
        ({"order_history": ["3_0"]}, 0.5),
        ({"order_history": ["3_0"]}, True),
        ({"order_history": ["3_0"]}, -10),
    ],
)
def test_validate_array_length_invalid_minimum_constraint(spark, raw_data, min_length):
    df = spark.createDataFrame([Row(**raw_data)])

    contract = {
        "schema": [
            {
                "name": "order_history",
                "type": "array<str>",
                "constraints": {"min_length": min_length, "max_length": 3},
            },
        ]
    }

    with pytest.raises(
        InvalidConstraintError, match="Minimum array length must be a positive integer"
    ):
        validate_array_lengths(df, contract=contract)


@pytest.mark.parametrize(
    ("raw_data", "max_length"),
    [
        ({"order_history": ["3_0"]}, 0.5),
        ({"order_history": ["3_0"]}, True),
        ({"order_history": ["3_0"]}, -10),
    ],
)
def test_validate_array_length_invalid_maximum_constraint(spark, raw_data, max_length):
    df = spark.createDataFrame([Row(**raw_data)])

    contract = {
        "schema": [
            {
                "name": "order_history",
                "type": "array<str>",
                "constraints": {"min_length": 1, "max_length": max_length},
            },
        ]
    }

    with pytest.raises(
        InvalidConstraintError, match="Maximum array length must be a positive integer"
    ):
        validate_array_lengths(df, contract=contract)


def test_validate_array_length_minimum_greater_than_maximum(spark):
    df = spark.createDataFrame([(["3_0", "2_3_4"],)], ["order_history"])

    contract = {
        "schema": [
            {
                "name": "order_history",
                "type": "array<str>",
                "constraints": {"min_length": 5, "max_length": 2},
            },
        ]
    }

    with pytest.raises(
        InvalidConstraintError, match="min_length cannot be greater than max_length"
    ):
        validate_array_lengths(df, contract=contract)


def test_validate_array_length_empty_dataframe(spark):
    schema = StructType([StructField("order_history", ArrayType(StringType()), False)])
    df = spark.createDataFrame([], schema=schema)

    contract = {
        "schema": [
            {
                "name": "order_history",
                "type": "array<str>",
                "constraints": {"min_length": 2, "max_length": 5},
            },
        ]
    }

    result = validate_array_lengths(df, contract=contract)[0]

    assert result.passed is True


def test_validate_array_length_invalid_rows_limit(spark):
    df = spark.createDataFrame(
        [([i for i in range(k)],) for k in range(50)], ["product_history"]
    )

    contract = {
        "schema": [
            {
                "name": "product_history",
                "type": "array<int>",
                "constraints": {"min_length": 50},
            },
        ]
    }

    result = validate_array_lengths(df, contract=contract)[0]

    assert result.passed is False
    assert result.failed_count == 50
    assert result.invalid_rows.count() == 30
    invalid_lengths = [
        len(row["product_history"]) for row in result.invalid_rows.collect()
    ]
    assert set(invalid_lengths).issubset(set(range(50)))
