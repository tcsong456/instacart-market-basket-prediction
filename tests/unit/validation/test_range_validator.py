from instacart_etl.validation.range import validate_range
from instacart_etl.validation.exceptions import InvalidConstraintError
import pytest

def test_range_validator_all_values_in_range(spark):

    df = spark.createDataFrame(
        [
            (1,),
            (5,),
            (10,),
        ],
        ['value']
    )

    result = validate_range(df, column_name='value', min_value=1, max_value=10)

    assert result.rule_name == 'value.range'
    assert result.category == 'range'
    assert result.passed is True
    assert result.failed_count == 0
    assert result.invalid_rows is None
    assert result.metadata == {
        'column_name': 'value',
        'minimum': 1,
        'maximum': 10
    }
    assert result.message == "column 'value' must be 1 and 10"


def test_range_validator_values_outside_range(spark):
    df = spark.createDataFrame(
        [
            (0,),
            (1,),
            (5,),
            (10,),
            (11,),
        ],
        ["value"],
    )

    result = validate_range(
        df,
        column_name="value",
        min_value=1,
        max_value=10,
    )

    assert result.passed is False
    assert result.failed_count == 2
    assert result.invalid_rows is not None

    invalid_values = {
        row["value"]
        for row in result.invalid_rows.collect()
    }

    assert invalid_values == {0, 11}


def test_range_validator_ignores_null_values(spark):
    df = spark.createDataFrame(
        [
            (None,),
            (5,),
        ],
        ["value"],
    )

    result = validate_range(
        df,
        column_name="value",
        min_value=1,
        max_value=10,
    )

    assert result.passed is True
    assert result.failed_count == 0
    assert result.invalid_rows is None


def test_range_validator_with_minimum_only(spark):
    df = spark.createDataFrame(
        [
            (0,),
            (1,),
            (100,),
        ],
        ["value"],
    )

    result = validate_range(
        df,
        column_name="value",
        min_value=1,
        max_value=None,
    )

    assert result.passed is False
    assert result.failed_count == 1
    assert result.metadata["minimum"] == 1
    assert result.metadata["maximum"] is None

    invalid_values = [
        row["value"]
        for row in result.invalid_rows.collect()
    ]

    assert invalid_values == [0]


def test_range_validator_with_maximum_only(spark):
    df = spark.createDataFrame(
        [
            (-100,),
            (10,),
            (11,),
        ],
        ["value"],
    )

    result = validate_range(
        df,
        column_name="value",
        min_value=None,
        max_value=10,
    )

    assert result.passed is False
    assert result.failed_count == 1
    assert result.metadata['minimum'] is None
    assert result.metadata['maximum'] == 10

    invalid_values = [
        row["value"]
        for row in result.invalid_rows.collect()
    ]

    assert invalid_values == [11]


def test_range_validator_with_no_boundary(spark):
    df = spark.createDataFrame(
        [
            (1,)
        ],
        ['value']
    )

    with pytest.raises(
        InvalidConstraintError,
        match='At least one of minimum'
    ):
        validate_range(
            df,
            column_name='value',
            min_value=None,
            max_value=None)


def test_range_validator_raises_when_minimum_exceeds_maximum(spark):
    df = spark.createDataFrame(
        [(1,)],
        ["value"],
    )

    with pytest.raises(
        InvalidConstraintError,
        match="minimum value should not be greater",
    ):
        validate_range(
            df,
            column_name="value",
            min_value=10,
            max_value=1,
        )


def test_range_validator_limits_invalid_row_sample_to_30(spark):
    df = spark.createDataFrame(
        [
            (value,)
            for value in range(50)
        ],
        ['value']
    )

    result = validate_range(
        df,
        column_name='value',
        min_value=100,
        max_value=None
    )

    assert result.passed is False
    assert result.failed_count == 50
    assert result.invalid_rows is not None
    assert result.invalid_rows.count() == 30