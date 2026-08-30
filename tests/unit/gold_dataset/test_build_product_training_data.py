from instacart_etl_rnn.gold.create_product_training_data import (
    build_product_training_data,
)


def test_build_product_training_data_pads_and_truncates_sequences(
    spark,
):
    product_history_data = spark.createDataFrame(
        [
            (
                10,
                1,
                "train",
                1,
                "1 0 1",
                "2 0 3",
                "3 4 5",
                "0 1 2",
                "10 11 12",
                "-1.0 7.0 5.0",
                "0 2 3",
                "1 2 3",
            ),
            (
                20,
                2,
                "test",
                -1,
                "1 1 0 1 1 0",
                "1 2 0 3 4 0",
                "5 6 4 7 8 3",
                "2 3 4 5 6 0",
                "15 16 17 18 19 20",
                "-1.0 3.0 4.0 5.0 6.0 7.0",
                "0 1 2 3 4 1",
                "1 2 3 4 5 6",
            ),
        ],
        """
        user_id int,
        product_id int,
        eval_set string,
        label int,
        is_ordered_history string,
        position_in_order_history string,
        history_order_size string,
        order_dows string,
        order_hours string,
        days_since_prior_orders string,
        history_reorder_size string,
        order_numbers string
        """,
    )

    encoded_product_name = spark.createDataFrame(
        [
            (1, "4 8"),
            (2, "9 7 6 5 4"),
        ],
        """
        product_id int,
        product_name_encoded string
        """,
    )

    result = build_product_training_data(
        product_history_data,
        encoded_product_name,
        product_name_length=4,
        encode_length=4,
    )

    actual = {row["product_id"]: row.asDict(recursive=True) for row in result.collect()}

    assert actual[1]["product_name_encoded"] == [4, 8, 0, 0]
    assert actual[1]["product_name_length"] == 2

    assert actual[1]["is_ordered_history"] == [1, 0, 1, 0]
    assert actual[1]["history_length"] == 3
    assert actual[1]["position_in_order_history"] == [2, 0, 3, 0]
    assert actual[1]["history_order_size"] == [3, 4, 5, 0]
    assert actual[1]["history_reorder_size"] == [0, 2, 3, 0]
    assert actual[1]["order_dows"] == [0, 1, 2, 0]
    assert actual[1]["order_hours"] == [10, 11, 12, 0]
    assert actual[1]["days_since_prior_orders"] == [-1.0, 7.0, 5.0, 0.0]
    assert actual[1]["order_numbers"] == [1, 2, 3, 0]

    assert actual[2]["product_name_encoded"] == [9, 7, 6, 5]
    assert actual[2]["product_name_length"] == 4

    assert actual[2]["is_ordered_history"] == [1, 1, 0, 1]
    assert actual[2]["history_length"] == 4
    assert actual[2]["position_in_order_history"] == [1, 2, 0, 3]
    assert actual[2]["history_order_size"] == [5, 6, 4, 7]
    assert actual[2]["history_reorder_size"] == [0, 1, 2, 3]
    assert actual[2]["order_dows"] == [2, 3, 4, 5]
    assert actual[2]["order_hours"] == [15, 16, 17, 18]
    assert actual[2]["days_since_prior_orders"] == [-1.0, 3.0, 4.0, 5.0]
    assert actual[2]["order_numbers"] == [1, 2, 3, 4]


def test_build_product_training_data_pads_missing_product_name_with_zeros(
    spark,
):
    product_history_data = spark.createDataFrame(
        [
            (
                10,
                99,
                "train",
                0,
                "1 0",
                "1 0",
                "2 3",
                "1 2",
                "10 11",
                "-1 5",
                "0 1",
                "1 2",
            ),
        ],
        """
        user_id int,
        product_id int,
        eval_set string,
        label int,
        is_ordered_history string,
        position_in_order_history string,
        history_order_size string,
        order_dows string,
        order_hours string,
        days_since_prior_orders string,
        history_reorder_size string,
        order_numbers string
        """,
    )

    encoded_product_name = spark.createDataFrame(
        [],
        """
        product_id int,
        product_name_encoded string
        """,
    )

    result = build_product_training_data(
        product_history_data,
        encoded_product_name,
        product_name_length=3,
        encode_length=3,
    )

    row = result.first()

    assert row["product_name_encoded"] == [0, 0, 0]
    assert row["product_name_length"] == 0

    assert result.columns == [
        "user_id",
        "product_id",
        "eval_set",
        "label",
        "product_name_encoded",
        "is_ordered_history",
        "position_in_order_history",
        "history_order_size",
        "history_reorder_size",
        "order_dows",
        "order_hours",
        "days_since_prior_orders",
        "order_numbers",
        "history_length",
        "product_name_length",
    ]
