
from instacart_etl_rnn.gold.create_product_history_data import parse_seq


def test_parse_seq_without_set(spark):
    df = spark.createDataFrame(
        [
            (1, "1_2 3_4 5_6"),
            (2, "7_8"),
        ],
        ["id", "seq"],
    )

    result = parse_seq(
        df,
        input_col="seq",
        prefix="product",
        compute_set=False,
    )

    actual = [
        (
            row.id,
            row.product_raw,
            row.product_prev,
            row.product_all,
            row.product_next,
            row.next_product_int,
        )
        for row in result.orderBy("id").collect()
    ]

    assert actual == [
        (
            1,
            ["1_2", "3_4", "5_6"],
            ["1_2", "3_4"],
            [[1, 2], [3, 4]],
            ["5_6"],
            [5, 6],
        ),
        (
            2,
            ["7_8"],
            ["7_8"],
            [[7, 8]],
            [],
            [],
        ),
    ]

    assert "product_set" not in result.columns
    assert "next_product_set" not in result.columns


def test_parse_seq_with_set(spark):
    df = spark.createDataFrame(
        [
            (
                1,
                "1_2_3 2_3_4 3_4_5",
            ),
            (
                2,
                "6_6_7",
            ),
        ],
        ["id", "seq"],
    )

    result = parse_seq(
        df,
        input_col="seq",
        prefix="product",
        compute_set=True,
    )

    rows = {row.id: row for row in result.collect()}

    assert rows[1]["product_all"] == [
        [1, 2, 3],
        [2, 3, 4],
    ]
    assert rows[1]["product_next"] == ["3_4_5"]
    assert rows[1]["next_product_int"] == [3, 4, 5]

    assert set(rows[1]["product_set"]) == {1, 2, 3, 4}
    assert set(rows[1]["next_product_set"]) == {3, 4, 5}

    assert rows[2]["product_all"] == [[6, 6, 7]]
    assert rows[2]["product_next"] == []
    assert rows[2]["next_product_int"] == []

    assert set(rows[2]["product_set"]) == {6, 7}
    assert rows[2]["next_product_set"] == []


def test_parse_seq_single_element_is_treated_as_previous(spark):
    df = spark.createDataFrame(
        [
            (1, "10_20_30"),
        ],
        ["id", "seq"],
    )

    result = parse_seq(
        df,
        input_col="seq",
        prefix="product",
        compute_set=True,
    )

    row = result.first()

    assert row.product_raw == ["10_20_30"]
    assert row.product_prev == ["10_20_30"]
    assert row.product_all == [[10, 20, 30]]

    assert row.product_next == []
    assert row.next_product_int == []

    assert row.product_set == [10, 20, 30]
    assert row.next_product_set == []


def test_parse_seq_reorders_with_set(spark):
    df = spark.createDataFrame(
        [
            (
                1,
                "1_0_1 0_1_1 1_1_0",
            ),
            (
                2,
                "1_1_1",
            ),
            (
                3,
                "0_0_0 0_0_0",
            ),
        ],
        ["id", "reorders"],
    )

    result = parse_seq(
        df,
        input_col="reorders",
        prefix="reorder",
        compute_set=False,
    )

    rows = {row.id: row for row in result.collect()}

    assert rows[1]["reorder_raw"] == [
        "1_0_1",
        "0_1_1",
        "1_1_0",
    ]
    assert rows[1]["reorder_prev"] == [
        "1_0_1",
        "0_1_1",
    ]
    assert rows[1]["reorder_all"] == [
        [1, 0, 1],
        [0, 1, 1],
    ]
    assert rows[1]["reorder_next"] == ["1_1_0"]
    assert rows[1]["next_reorder_int"] == [1, 1, 0]

    assert rows[2]["reorder_raw"] == ["1_1_1"]
    assert rows[2]["reorder_prev"] == ["1_1_1"]
    assert rows[2]["reorder_all"] == [[1, 1, 1]]
    assert rows[2]["reorder_next"] == []
    assert rows[2]["next_reorder_int"] == []

    assert rows[3]["reorder_all"] == [[0, 0, 0]]
    assert rows[3]["reorder_next"] == ["0_0_0"]
    assert rows[3]["next_reorder_int"] == [0, 0, 0]
