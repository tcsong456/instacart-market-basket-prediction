from instacart_etl_rnn.gold.create_product_history_data import (
    build_each_product_in_order_history,
)


def test_build_each_product_in_order_history(mocker, spark):
    input_df = spark.createDataFrame(
        [
            (
                1,
                [[1, 2, 10], [2, 3, 5], [2, 5, 11], [10]],
                [1, 2, 3, 5, 10, 11],
                [2, 3],
                "3 2 6 6 1",
                "18 10 3 9 20",
                "27 7 11 23 30",
                "1 2 3 4 5",
                "train",
            ),
        ],
        [
            "user_id",
            "products_all",
            "products_set",
            "next_products_set",
            "order_dows",
            "order_hours",
            "days_since_prior_orders",
            "order_numbers",
            "eval_set",
        ],
    )

    products_df = spark.createDataFrame(
        [
            (1, "apple", 3, 30),
            (2, "banana", 3, 30),
            (3, "orange", 2, 20),
            (5, "grape", 1, 10),
            (10, "peach", 2, 20),
            (11, "pear", 1, 10),
        ],
        [
            "product_id",
            "product_name",
            "aisle_id",
            "department_id",
        ],
    )

    mock_join_path = mocker.patch(
        "instacart_etl_rnn.gold.create_product_history_data.join_path",
        return_value="gs://bucket/products",
    )

    mock_read_parquet = mocker.patch(
        "instacart_etl_rnn.gold.create_product_history_data.read_parquet",
        return_value=products_df,
    )

    result = build_each_product_in_order_history(
        path="gs://bucket",
        df=input_df,
        spark=spark,
    )

    rows = {row.product_id: row for row in result.collect()}

    assert set(rows) == {1, 2, 3, 5, 10, 11}

    for row in rows.values():
        assert row.history_order_size == "3 3 3 1"
        assert row.history_reorder_size == "0 1 2 1"

    assert rows[1].label == 0
    assert rows[1].is_ordered_history == "1 0 0 0"
    assert rows[1].eval_set == "train"
    assert rows[1].position_in_order_history == "1 0 0 0"
    assert rows[1].product_name == "apple"
    assert rows[1].aisle_id == 3
    assert rows[1].department_id == 30

    assert rows[2].label == 1
    assert rows[2].is_ordered_history == "1 1 1 0"
    assert rows[2].position_in_order_history == "2 1 1 0"
    assert rows[2].product_name == "banana"

    assert rows[3].label == 1
    assert rows[3].is_ordered_history == "0 1 0 0"
    assert rows[3].position_in_order_history == "0 2 0 0"
    assert rows[3].product_name == "orange"

    assert rows[5].label == 0
    assert rows[5].is_ordered_history == "0 1 1 0"
    assert rows[5].position_in_order_history == "0 3 2 0"
    assert rows[5].product_name == "grape"

    assert rows[10].label == 0
    assert rows[10].is_ordered_history == "1 0 0 1"
    assert rows[10].position_in_order_history == "3 0 0 1"
    assert rows[10].product_name == "peach"

    assert rows[11].label == 0
    assert rows[11].is_ordered_history == "0 0 1 0"
    assert rows[11].position_in_order_history == "0 0 3 0"
    assert rows[11].product_name == "pear"

    assert rows[1].order_dows == "3 2 6 6 1"
    assert rows[1].order_hours == "18 10 3 9 20"
    assert rows[1].days_since_prior_orders == "27 7 11 23 30"
    assert rows[1].order_numbers == "1 2 3 4 5"

    mock_join_path.assert_called_once_with(
        "gs://bucket",
        "products",
    )

    mock_read_parquet.assert_called_once_with(
        "gs://bucket/products",
        spark,
    )


def test_build_each_product_in_order_history_empty_next_basket(
    mocker,
    spark,
):
    input_df = spark.createDataFrame(
        [
            (1, [[1, 2]], [1, 2], [], "5", "19", "16", "1", "prior"),
        ],
        schema="""
            user_id INT,
            products_all ARRAY<ARRAY<INT>>,
            products_set ARRAY<INT>,
            next_products_set ARRAY<INT>,
            order_dows STRING,
            order_hours STRING,
            days_since_prior_orders STRING,
            order_numbers STRING,
            eval_set STRING
        """,
    )

    products_df = spark.createDataFrame(
        [
            (1, "apple", 3, 30),
            (2, "banana", 3, 30),
        ],
        [
            "product_id",
            "product_name",
            "aisle_id",
            "department_id",
        ],
    )

    mocker.patch(
        "instacart_etl_rnn.gold.create_product_history_data.join_path",
        return_value="gs://bucket/products",
    )

    mocker.patch(
        "instacart_etl_rnn.gold.create_product_history_data.read_parquet",
        return_value=products_df,
    )

    result = build_each_product_in_order_history(
        path="gs://bucket",
        df=input_df,
        spark=spark,
    )

    rows = {row.product_id: row for row in result.collect()}

    assert set(rows) == {1, 2}

    assert rows[1].history_order_size == "2"
    assert rows[1].history_reorder_size == "0"
    assert rows[1].label == 0
    assert rows[1].eval_set == "prior"
    assert rows[1].is_ordered_history == "1"
    assert rows[1].position_in_order_history == "1"
    assert rows[1].product_name == "apple"

    assert rows[2].history_order_size == "2"
    assert rows[2].history_reorder_size == "0"
    assert rows[2].label == 0
    assert rows[2].is_ordered_history == "1"
    assert rows[2].position_in_order_history == "2"
    assert rows[2].product_name == "banana"
