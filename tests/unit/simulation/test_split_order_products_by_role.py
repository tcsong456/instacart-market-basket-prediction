from instacart_etl_rnn.simulation.create_order_product_split import (
    split_order_products_by_role,
)


def test_split_order_products_by_role(spark):
    df = spark.createDataFrame(
        [
            (1, True, False, True),
            (2, True, False, False),
            (3, False, True, False),
            (4, False, False, True),
            (5, False, False, False),
        ],
        """
        order_id int,
        is_train_available boolean,
        is_evaluation_available boolean,
        is_validation_available boolean
        """,
    )

    train, evaluation, validation = split_order_products_by_role(df)

    assert {row.order_id for row in train.collect()} == {1, 2}

    assert {row.order_id for row in evaluation.collect()} == {3}

    assert {row.order_id for row in validation.collect()} == {1, 4}
