from instacart_etl_rnn.gold.create_product_history_data import (
    build_each_reorder_history,
)


def test_build_each_reorder_history(spark):
    df = spark.createDataFrame(
        [
            (
                1,
                [[0, 0], [1, 0], [1, 1, 0]],
                [0, 0],
                "3 2 6 6",
                "18 10 3 9",
                "27 7 11 23",
                "1 2 3 4",
            ),
            (2, [[0, 0], [0, 0]], [1, 0], "0 2 1", "5 17 22", "30 24 28", "1 2 3"),
            (
                3,
                [[1, 0], [1, 1]],
                [0, 0],
                "4 0 1",
                "13 21 13",
                "10 6 15",
                "1 2 3",
            ),
        ],
        [
            "user_id",
            "reorders_all",
            "next_reorders_int",
            "order_dows",
            "order_hours",
            "days_since_prior_orders",
            "order_numbers",
        ],
    )
    
    result = build_each_reorder_history(df)

    rows = {row.user_id: row for row in result.collect()}

    assert rows[1]["label"] == 1

    assert rows[1]["is_ordered_history"] == "1 0 0"
    assert rows[1]["position_in_order_history"] == "0 0 0"
    assert rows[1]["order_dows"] == "3 2 6 6"
    assert rows[1]["order_hours"] == "18 10 3 9"
    assert rows[1]["days_since_prior_orders"] == "27 7 11 23"
    assert rows[1]["order_numbers"] == "1 2 3 4"
    assert rows[1]["history_order_size"] == "2 2 3"
    assert rows[1]["history_reorder_size"] == "0 1 2"

    assert rows[2]["label"] == 0

    assert rows[2]["is_ordered_history"] == "1 1"
    assert rows[2]["position_in_order_history"] == "0 0"
    assert rows[2]["history_order_size"] == "2 2"
    assert rows[2]["history_reorder_size"] == "0 0"

    assert rows[3]["label"] == -1

    assert rows[3]["is_ordered_history"] == "0 0"
    assert rows[3]["position_in_order_history"] == "0 0"
    assert rows[3]["history_order_size"] == "2 2"
    assert rows[3]["history_reorder_size"] == "1 2"

    for row in rows.values():
        assert row["product_id"] == 0
        assert row["aisle_id"] == 0
        assert row["department_id"] == 0
        assert row["product_name"] == ""
