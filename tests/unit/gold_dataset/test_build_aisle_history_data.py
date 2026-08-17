from instacart_etl_rnn.gold.create_aisle_history_data import build_aisle_history_data


def test_build_aisle_history_data_builds_expected_history(
    spark,
):
    df = spark.createDataFrame(
        [
            (
                1,
                [
                    [1, 1, 2],
                    [2, 3, 3],
                    [1, 3],
                ],
                [1, 2, 3],
            ),
        ],
        """
        user_id INT,
        aisle_all ARRAY<ARRAY<INT>>,
        aisle_set ARRAY<INT>
        """,
    )

    result = build_aisle_history_data(df)

    actual = {row.aisle_id: row.asDict(recursive=True) for row in result.collect()}

    assert set(actual) == {1, 2, 3}

    assert actual[1]["aisle_history_size"] == "2 2 2"
    assert actual[1]["is_ordered_history"] == "1 0 1"
    assert actual[1]["position_in_order"] == "1 0 1"
    assert actual[1]["num_products_from_aisle"] == "2 0 1"

    assert actual[2]["aisle_history_size"] == "2 2 2"
    assert actual[2]["is_ordered_history"] == "1 1 0"
    assert actual[2]["position_in_order"] == "2 1 0"
    assert actual[2]["num_products_from_aisle"] == "1 1 0"

    assert actual[3]["aisle_history_size"] == "2 2 2"
    assert actual[3]["is_ordered_history"] == "0 1 1"
    assert actual[3]["position_in_order"] == "0 2 2"
    assert actual[3]["num_products_from_aisle"] == "0 2 1"


def test_build_aisle_history_data_handles_single_aisle(
    spark,
):
    df = spark.createDataFrame(
        [
            (
                1,
                [[5, 5, 5]],
                [5],
            ),
        ],
        """
        user_id INT,
        aisle_all ARRAY<ARRAY<INT>>,
        aisle_set ARRAY<INT>
        """,
    )

    result = build_aisle_history_data(df)

    row = result.first()

    assert row.aisle_id == 5
    assert row.aisle_history_size == "1"
    assert row.is_ordered_history == "1"
    assert row.position_in_order == "1"
    assert row.num_products_from_aisle == "3"
