import pytest
from pyspark.sql import functions as F

from instacart_etl_rnn.common.utils import parse_string_sequence


def test_parse_string_sequence_parses_space_separated_integers(spark):
    df = spark.createDataFrame(
        [
            ("1 2 3",),
            ("10 20 30",),
        ],
        ["sequence"],
    )

    result = df.select(parse_string_sequence(F.col("sequence")).alias("parsed"))

    actual = [row["parsed"] for row in result.collect()]

    assert actual == [
        [1, 2, 3],
        [10, 20, 30],
    ]


def test_parse_string_sequence_handles_extra_whitespace(spark):
    df = spark.createDataFrame(
        [
            ("  1   2  3  ",),
        ],
        ["sequence"],
    )

    result = df.select(parse_string_sequence(F.col("sequence")).alias("parsed"))

    assert result.first()["parsed"] == [1, 2, 3]


def test_parse_string_sequence_returns_empty_array_for_blank_string(spark):
    df = spark.createDataFrame(
        [
            ("",),
            ("   ",),
        ],
        ["sequence"],
    )

    result = df.select(parse_string_sequence(F.col("sequence")).alias("parsed"))

    actual = [row["parsed"] for row in result.collect()]

    assert actual == [
        [],
        [],
    ]


def test_parse_string_sequence_returns_empty_array_for_null(spark):
    df = spark.createDataFrame(
        [(None,)],
        "sequence string",
    )

    result = df.select(parse_string_sequence(F.col("sequence")).alias("parsed"))

    assert result.first()["parsed"] == []


def test_parse_string_sequence_uses_custom_pattern(spark):
    df = spark.createDataFrame(
        [
            ("1,2,3",),
        ],
        ["sequence"],
    )

    result = df.select(
        parse_string_sequence(
            F.col("sequence"),
            pattern=",",
        ).alias("parsed")
    )

    assert result.first()["parsed"] == [1, 2, 3]


@pytest.mark.parametrize(
    ("data_type", "input_value", "expected"),
    [
        ("int", "1 2 3", [1, 2, 3]),
        ("double", "1.5 2.0 3.25", [1.5, 2.0, 3.25]),
        ("float", "1.5 2.0 3.25", [1.5, 2.0, 3.25]),
        ("bool", "true false true", [True, False, True]),
    ],
)
def test_parse_string_sequence_parses_supported_types(
    spark,
    data_type,
    input_value,
    expected,
):
    df = spark.createDataFrame(
        [(input_value,)],
        ["sequence"],
    )

    result = df.select(
        parse_string_sequence(
            F.col("sequence"),
            data_type=data_type,
        ).alias("parsed")
    )

    assert result.first()["parsed"] == expected


def test_parse_string_sequence_raises_for_unsupported_type(spark):
    with pytest.raises(
        ValueError,
        match="string is not supported!",
    ):
        parse_string_sequence(
            F.col("sequence"),
            data_type="string",
        )
