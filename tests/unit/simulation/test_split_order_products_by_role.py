from instacart_etl_rnn.simulation.create_order_product_split import (
    split_order_products_by_role,
)


def test_split_order_products_by_role_splits_expected_rows(spark):
    order_products = spark.createDataFrame(
        [
            (1, 101, "history"),
            (1, 102, "history"),
            (1, 103, "train_label"),
            (1, 104, "validation_label"),
            (1, 105, "future"),
            (2, 201, None),
        ],
        [
            "user_id",
            "order_id",
            "order_role",
        ],
    )

    history, train_label, validation_label = split_order_products_by_role(
        order_products
    )

    actual_history = {row["order_id"] for row in history.collect()}

    actual_train_label = {row["order_id"] for row in train_label.collect()}

    actual_validation_label = {row["order_id"] for row in validation_label.collect()}

    assert actual_history == {101, 102}
    assert actual_train_label == {103}
    assert actual_validation_label == {104}
