from instacart_etl_rnn.gold.create_reorder_size_training_data import (
    transform_reorder_size_data,
)


def test_transform_reorder_size_training_data_builds_expected_features(
    spark,
):
    df = spark.createDataFrame(
        [
            (
                1,
                "0_1_0 1_1_0 1_0_1",
            ),
        ],
        """
        user_id INT,
        reorders STRING
        """,
    )

    result = transform_reorder_size_data(df)

    row = result.first()

    assert row["reorders"] == [
        "0_1_0",
        "1_1_0",
        "1_0_1",
    ]

    assert row["reorders_prev"] == [
        [0, 1, 0],
        [1, 1, 0],
    ]

    assert row["reorders_next"] == [
        1,
        0,
        1,
    ]

    assert row["order_sizes"] == [
        3,
        3,
    ]

    assert row["reorder_sizes"] == [
        1,
        2,
    ]

    assert row["label"] == 2


def test_transform_reorder_size_training_data_handles_single_order(
    spark,
):
    df = spark.createDataFrame(
        [
            (
                1,
                "0_1_1",
            ),
        ],
        """
        user_id INT,
        reorders STRING
        """,
    )

    result = transform_reorder_size_data(df)

    row = result.first()

    assert row["reorders"] == [
        "0_1_1",
    ]

    assert row["reorders_prev"] == [
        [0, 1, 1],
    ]

    assert row["reorders_next"] == []

    assert row["order_sizes"] == [
        3,
    ]

    assert row["reorder_sizes"] == [
        2,
    ]

    assert row["label"] == 0


def test_transform_reorder_size_training_data_handles_null_and_blank_reorders(
    spark,
):
    df = spark.createDataFrame(
        [
            (1, None),
            (2, ""),
            (3, "   "),
        ],
        """
        user_id INT,
        reorders STRING
        """,
    )

    result = transform_reorder_size_data(df)

    actual = {row.user_id: row.asDict(recursive=True) for row in result.collect()}

    for user_id in [1, 2, 3]:
        row = actual[user_id]

        assert row["reorders"] == []
        assert row["reorders_prev"] == []
        assert row["reorders_next"] == []
        assert row["order_sizes"] == []
        assert row["reorder_sizes"] == []
        assert row["label"] == 0


def test_transform_reorder_size_training_data_calculates_sizes_correctly(
    spark,
):
    df = spark.createDataFrame(
        [
            (
                1,
                "0_1 1_0_1_1 1",
            ),
        ],
        """
        user_id INT,
        reorders STRING
        """,
    )

    result = transform_reorder_size_data(df)

    row = result.first()

    assert row["reorders_prev"] == [
        [0, 1],
        [1, 0, 1, 1],
    ]

    assert row["order_sizes"] == [
        2,
        4,
    ]

    assert row["reorder_sizes"] == [
        1,
        3,
    ]

    assert row["reorders_next"] == [
        1,
    ]

    assert row["label"] == 1
