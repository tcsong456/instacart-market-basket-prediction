from instacart_etl_rnn.gold.create_aisle_history_data import parse_seq


def test_parse_seq_parses_multiple_orders(spark):
    df = spark.createDataFrame(
        [
            (
                1,
                "1_2 2_3_4 5",
            ),
        ],
        """
        user_id INT,
        aisle_ids STRING
        """,
    )

    result = parse_seq(df)

    row = result.first()

    assert row["aisle_raw"] == ["1_2", "2_3_4", "5"]

    assert row["aisle_prev"] == ["1_2", "2_3_4"]

    assert row["aisle_all"] == [[1, 2], [2, 3, 4]]

    assert row["aisle_set"] == [1, 2, 3, 4]
    assert row["aisle_next"] == ["5"]
    assert row["next_aisle_int"] == [5]
    assert row["next_aisle_set"] == [5]


def test_parse_seq_handles_single_order(
    spark,
):
    df = spark.createDataFrame(
        [
            (
                1,
                "1_2_3",
            ),
        ],
        """
        user_id INT,
        aisle_ids STRING
        """,
    )

    result = parse_seq(df)

    row = result.first()

    assert row["aisle_raw"] == [
        "1_2_3",
    ]

    assert row["aisle_prev"] == [
        "1_2_3",
    ]

    assert row["aisle_all"] == [
        [1, 2, 3],
    ]

    assert row["aisle_set"] == [
        1,
        2,
        3,
    ]
    assert row["aisle_next"] == []
    assert row["next_aisle_int"] == []
    assert row["next_aisle_set"] == []


def test_parse_seq_removes_duplicate_aisle_ids(
    spark,
):
    df = spark.createDataFrame(
        [
            (
                1,
                "1_2_2 2_3_3 4_4_5",
            ),
        ],
        """
        user_id INT,
        aisle_ids STRING
        """,
    )

    result = parse_seq(df)

    row = result.first()

    assert row["aisle_all"] == [
        [1, 2, 2],
        [2, 3, 3],
    ]

    assert row["aisle_set"] == [
        1,
        2,
        3,
    ]
