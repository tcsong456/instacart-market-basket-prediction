from pyspark.sql import functions as F

from instacart_etl_rnn.common.utils import pad_array


def test_pad_array_pads_short_array(spark):
    df = spark.createDataFrame(
        [
            ([1, 2, 3],),
        ],
        "values array<int>",
    )

    padded, seq_length = pad_array(
        F.col("values"),
        max_length=5,
    )

    result = df.select(
        padded.alias("padded"),
        seq_length.alias("seq_length"),
    )

    row = result.first()

    assert row["padded"] == [1, 2, 3, 0, 0]
    assert row["seq_length"] == 3


def test_pad_array_keeps_array_at_max_length(spark):
    df = spark.createDataFrame(
        [
            ([1, 2, 3, 4, 5],),
        ],
        "values array<int>",
    )

    padded, seq_length = pad_array(
        F.col("values"),
        max_length=5,
    )

    result = df.select(
        padded.alias("padded"),
        seq_length.alias("seq_length"),
    )

    row = result.first()

    assert row["padded"] == [1, 2, 3, 4, 5]
    assert row["seq_length"] == 5


def test_pad_array_truncates_long_array(spark):
    df = spark.createDataFrame(
        [
            ([1, 2, 3, 4, 5, 6, 7],),
        ],
        "values array<int>",
    )

    padded, seq_length = pad_array(
        F.col("values"),
        max_length=5,
    )

    result = df.select(
        padded.alias("padded"),
        seq_length.alias("seq_length"),
    )

    row = result.first()

    assert row["padded"] == [1, 2, 3, 4, 5]
    assert row["seq_length"] == 5


def test_pad_array_pads_empty_array(spark):
    df = spark.createDataFrame(
        [
            ([],),
        ],
        "values array<int>",
    )

    padded, seq_length = pad_array(
        F.col("values"),
        max_length=4,
    )

    result = df.select(
        padded.alias("padded"),
        seq_length.alias("seq_length"),
    )

    row = result.first()

    assert row["padded"] == [0, 0, 0, 0]
    assert row["seq_length"] == 0
