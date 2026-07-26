import pytest
from pyspark.sql.types import IntegerType, StructField, StructType

from instacart_etl.validation.exceptions import InvalidConstraintError
from instacart_etl.validation.range import validate_range


def test_range_validator_all_values_in_range(spark):
    df = spark.createDataFrame([(1, 0.5), (5, 1.5), (10, 2.5)], ["sales", "price"])

    contract = {
        "schema": [
            {"name": "sales", "constraints": {"minimum": 1, "maximum": 10}},
            {"name": "price", "constraints": {"minimum": 0.5, "maximum": 2.5}},
        ]
    }

    results = validate_range(df, contract=contract)

    assert results[0].rule_name == "sales.range"
    assert results[0].category == "range"
    assert results[0].passed is True
    assert results[0].failed_count == 0
    assert results[0].invalid_rows is None
    assert results[0].message == "column 'sales' must be between 1 and 10"
    assert results[0].metadata == {
        "column_name": "sales",
        "minimum": 1,
        "maximum": 10,
    }

    assert results[1].rule_name == "price.range"
    assert results[1].category == "range"
    assert results[1].passed is True
    assert results[1].failed_count == 0
    assert results[1].invalid_rows is None
    assert results[1].message == "column 'price' must be between 0.5 and 2.5"
    assert results[1].metadata == {
        "column_name": "price",
        "minimum": 0.5,
        "maximum": 2.5,
    }


def test_range_validator_values_outside_range(spark):
    df = spark.createDataFrame([(1, 0.5), (5, 1.5), (10, 2.5)], ["sales", "price"])

    contract = {
        "schema": [
            {"name": "sales", "constraints": {"minimum": 2, "maximum": 10}},
            {"name": "price", "constraints": {"minimum": 1.0, "maximum": 2.0}},
        ]
    }

    results = validate_range(df, contract=contract)

    assert results[0].passed is False
    assert results[0].failed_count == 1
    invalid_rows = {row["sales"] for row in results[0].invalid_rows.collect()}
    assert invalid_rows == {1}

    assert results[1].passed is False
    assert results[1].failed_count == 2
    invalid_rows = {row["price"] for row in results[1].invalid_rows.collect()}
    assert invalid_rows == {0.5, 2.5}


def test_range_validator_ignores_null_values(spark):
    df = spark.createDataFrame(
        [(None, 0.5), (5, None), (10, 1.0), (None, 2.5)],
        ["sales", "price"],
    )

    contract = {
        "schema": [
            {"name": "sales", "constraints": {"minimum": 1, "maximum": 10}},
            {"name": "price", "constraints": {"minimum": 1.0, "maximum": 3.0}},
        ]
    }

    results = validate_range(df, contract=contract)

    assert results[0].passed is True
    assert results[0].failed_count == 0

    assert results[1].passed is False
    assert results[1].failed_count == 1


def test_range_validator_with_minimum_only(spark):
    df = spark.createDataFrame(
        [
            (0,),
            (1,),
            (100,),
        ],
        ["sales"],
    )

    contract = {"schema": [{"name": "sales", "constraints": {"minimum": 1}}]}

    result = validate_range(df, contract=contract)[0]

    assert result.passed is False
    assert result.failed_count == 1
    assert result.metadata["minimum"] == 1
    assert result.metadata["maximum"] is None

    invalid_values = [row["sales"] for row in result.invalid_rows.collect()]

    assert invalid_values == [0]


def test_range_validator_with_maximum_only(spark):
    df = spark.createDataFrame(
        [
            (-100,),
            (10,),
            (11,),
        ],
        ["sales"],
    )

    contract = {"schema": [{"name": "sales", "constraints": {"maximum": 10}}]}

    result = validate_range(df, contract=contract)[0]

    assert result.passed is False
    assert result.failed_count == 1
    assert result.metadata["minimum"] is None
    assert result.metadata["maximum"] == 10

    invalid_values = [row["sales"] for row in result.invalid_rows.collect()]

    assert invalid_values == [11]


def test_range_validator_with_no_boundary(spark):
    df = spark.createDataFrame(
        [
            (1,),
            (5,),
            (10,),
        ],
        ["sales"],
    )

    contract = {"schema": [{"name": "sales"}]}

    result = validate_range(df, contract=contract)

    assert result == []


def test_range_validator_raises_when_minimum_exceeds_maximum(spark):
    df = spark.createDataFrame(
        [(1,)],
        ["sales"],
    )

    contract = {
        "schema": [{"name": "sales", "constraints": {"minimum": 10, "maximum": 1}}]
    }

    with pytest.raises(
        InvalidConstraintError,
        match="Minimum for column 'sales'",
    ):
        validate_range(df, contract=contract)


def test_range_validator_raises_when_minimum_is_invalid(spark):
    df = spark.createDataFrame(
        [(1,)],
        ["sales"],
    )

    contract = {
        "schema": [
            {"name": "sales", "constraints": {"minimum": "false", "maximum": 10}}
        ]
    }

    with pytest.raises(
        InvalidConstraintError,
        match="Minimum for column 'sales' must be a valid number",
    ):
        validate_range(df, contract=contract)


def test_range_validator_raises_when_maximum_is_invalid(spark):
    df = spark.createDataFrame(
        [(1,)],
        ["sales"],
    )

    contract = {
        "schema": [{"name": "sales", "constraints": {"minimum": 1, "maximum": "true"}}]
    }

    with pytest.raises(
        InvalidConstraintError,
        match="Maximum for column 'sales' must be a valid number",
    ):
        validate_range(df, contract=contract)


def test_range_validator_limits_invalid_row_sample_to_30(spark):
    df = spark.createDataFrame([(value,) for value in range(50)], ["sales"])

    contract = {"schema": [{"name": "sales", "constraints": {"minimum": 50}}]}

    result = validate_range(df, contract=contract)[0]

    assert result.passed is False
    assert result.failed_count == 50
    assert result.invalid_rows is not None
    assert result.invalid_rows.count() == 30


def test_range_validator_when_dataframe_is_empty(spark):
    schema = StructType([StructField("sales", IntegerType(), False)])
    df = spark.createDataFrame([], schema=schema)

    contract = {
        "schema": [{"name": "sales", "constraints": {"minimum": 1, "maximum": 10}}]
    }

    result = validate_range(df, contract=contract)[0]

    assert result.passed is True
    assert result.failed_count == 0
    assert result.invalid_rows is None


def test_range_validator_with_all_values_null(spark):
    schema = StructType([StructField("sales", IntegerType(), True)])
    df = spark.createDataFrame(
        [
            (None,),
            (None,),
            (None,),
        ],
        schema=schema,
    )

    contract = {"schema": [{"name": "sales", "constraints": {"minimum": 1}}]}

    result = validate_range(df, contract=contract)[0]

    assert result.passed is True
    assert result.failed_count == 0
