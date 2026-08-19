from instacart_etl_rnn.gold.create_reorder_size_training_data import pad_column_arrays


def test_pad_column_arrays_parses_and_pads_with_next_order_metadata(
    spark,
):
    df = spark.createDataFrame(
        [
            (
                1,
                "train",
                [3, 2],
                [1, 1],
                2,
                "1 2 3",
                "10 11 12",
                "-1.0 5.0 7.0",
                "1 2 3",
            ),
        ],
        """
        user_id INT,
        eval_set STRING,
        order_sizes ARRAY<INT>,
        reorder_sizes ARRAY<INT>,
        label INT,
        order_dows STRING,
        order_hours STRING,
        days_since_prior_orders STRING,
        order_numbers STRING
        """,
    )

    result = pad_column_arrays(
        df,
        pad_length=4,
    )

    row = result.first()

    assert row["user_id"] == 1
    assert row["eval_set"] == "train"
    assert row["label"] == 2

    assert row["order_sizes"] == [
        3,
        2,
        0,
        0,
    ]

    assert row["reorder_sizes"] == [
        1,
        1,
        0,
        0,
    ]

    assert row["order_dows"] == [
        1,
        2,
        3,
        0,
    ]

    assert row["order_hours"] == [
        10,
        11,
        12,
        0,
    ]

    assert row["days_since_prior_orders"] == [
        -1.0,
        5.0,
        7.0,
        0.0,
    ]

    assert row["order_numbers"] == [
        1,
        2,
        3,
        0,
    ]

    assert row["history_length"] == 2

    assert result.columns == [
        "user_id",
        "eval_set",
        "order_sizes",
        "reorder_sizes",
        "label",
        "order_dows",
        "order_hours",
        "days_since_prior_orders",
        "order_numbers",
        "history_length",
    ]


def test_pad_column_arrays_handles_no_previous_orders(
    spark,
):
    df = spark.createDataFrame(
        [
            (
                1,
                "test",
                [3],
                [1],
                0,
                "2",
                "15",
                "-1.0",
                "1",
            ),
        ],
        """
        user_id INT,
        eval_set STRING,
        order_sizes ARRAY<INT>,
        reorder_sizes ARRAY<INT>,
        label INT,
        order_dows STRING,
        order_hours STRING,
        days_since_prior_orders STRING,
        order_numbers STRING
        """,
    )

    result = pad_column_arrays(
        df,
        pad_length=3,
    )

    row = result.first()

    assert row["order_sizes"] == [3, 0, 0]
    assert row["reorder_sizes"] == [1, 0, 0]

    assert row["order_dows"] == [2, 0, 0]
    assert row["order_hours"] == [15, 0, 0]
    assert row["days_since_prior_orders"] == [
        -1.0,
        0.0,
        0.0,
    ]
    assert row["order_numbers"] == [1, 0, 0]

    assert row["history_length"] == 1


def test_pad_column_arrays_keeps_arrays_at_pad_length(
    spark,
):
    df = spark.createDataFrame(
        [
            (
                1,
                "train",
                [3, 2, 4],
                [1, 1, 2],
                1,
                "2 4 3 6",
                "10 11 12 13",
                "-1.0 5.0 7.0 11.0",
                "1 2 3 4",
            ),
        ],
        """
        user_id INT,
        eval_set STRING,
        order_sizes ARRAY<INT>,
        reorder_sizes ARRAY<INT>,
        label INT,
        order_dows STRING,
        order_hours STRING,
        days_since_prior_orders STRING,
        order_numbers STRING
        """,
    )

    result = pad_column_arrays(
        df,
        pad_length=3,
    )

    row = result.first()

    assert row["order_sizes"] == [3, 2, 4]
    assert row["reorder_sizes"] == [1, 1, 2]
    assert row["order_dows"] == [2, 4, 3]
    assert row["order_hours"] == [10, 11, 12]
    assert row["days_since_prior_orders"] == [
        -1.0,
        5.0,
        7.0,
    ]
    assert row["order_numbers"] == [1, 2, 3]

    assert row["history_length"] == 3


def test_pad_column_arrays_truncates_arrays_longer_than_pad_length(
    spark,
):
    df = spark.createDataFrame(
        [
            (
                1,
                "train",
                [1, 2, 3, 4],
                [0, 1, 1, 2],
                1,
                "1 2 3 4 5",
                "10 11 12 13 14",
                "-1.0 5.0 7.0 8.0 10.0",
                "1 2 3 4 5",
            ),
        ],
        """
        user_id INT,
        eval_set STRING,
        order_sizes ARRAY<INT>,
        reorder_sizes ARRAY<INT>,
        label INT,
        order_dows STRING,
        order_hours STRING,
        days_since_prior_orders STRING,
        order_numbers STRING
        """,
    )

    result = pad_column_arrays(
        df,
        pad_length=3,
    )

    row = result.first()

    assert len(row["order_sizes"]) == 3
    assert len(row["reorder_sizes"]) == 3
    assert len(row["order_dows"]) == 3
    assert len(row["order_hours"]) == 3
    assert len(row["days_since_prior_orders"]) == 3
    assert len(row["order_numbers"]) == 3

    assert row["history_length"] == 3
