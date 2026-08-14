from instacart_etl_rnn.gold.create_product_history_data import filtered_orders


def test_filtered_orders(mocker, spark):
    orders = spark.createDataFrame(
        [
            (1, "prior"),
            (1, "train"),
            (2, "test"),
            (3, "prior"),
        ],
        ["user_id", "eval_set"],
    )

    expected_df = spark.createDataFrame(
        [
            (1, "train"),
            (2, "test"),
        ],
        ["user_id", "train_eval_set"],
    )

    mock_join_path = mocker.patch(
        "instacart_etl_rnn.gold.create_product_history_data.join_path",
        return_value="gs://bucket/orders",
    )
    mock_read_parquet = mocker.patch(
        "instacart_etl_rnn.gold.create_product_history_data.read_parquet",
        return_value=orders,
    )

    actual_df = filtered_orders(
        path="gs://bucket",
        spark=spark,
    )

    assert actual_df.schema == expected_df.schema

    assert set(actual_df.collect()) == set(expected_df.collect())

    mock_join_path.assert_called_once_with(
        "gs://bucket",
        "orders",
    )
    mock_read_parquet.assert_called_once_with(
        "gs://bucket/orders",
        spark,
    )


def test_filtered_orders_only_contains_prior(mocker, spark):
    orders = spark.createDataFrame(
        [
            (1, "prior"),
            (3, "prior"),
        ],
        ["user_id", "eval_set"],
    )

    mocker.patch(
        "instacart_etl_rnn.gold.create_product_history_data.join_path",
        return_value="gs://bucket/orders",
    )
    mocker.patch(
        "instacart_etl_rnn.gold.create_product_history_data.read_parquet",
        return_value=orders,
    )

    actual_df = filtered_orders(
        path="gs://bucket",
        spark=spark,
    )

    assert actual_df.isEmpty()

    assert actual_df.columns == ["user_id", "train_eval_set"]
