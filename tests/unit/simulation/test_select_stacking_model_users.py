from instacart_etl_rnn.simulation.create_order_product_split import (
    select_stacking_model_users,
)


def test_select_stacking_model_users_filters_and_assigns_roles(spark):
    df = spark.createDataFrame(
        [
            (1, 1, 4, "established", "stacking_train"),
            (1, 2, 4, "established", "stacking_train"),
            (1, 3, 4, "established", "stacking_train"),
            (1, 4, 4, "established", "stacking_train"),
            (2, 1, 3, "established", "base_train"),
            (2, 2, 3, "established", "base_train"),
            (2, 3, 3, "established", "base_train"),
            (3, 1, 3, "new_user", "stacking_train"),
            (3, 2, 3, "new_user", "stacking_train"),
            (3, 3, 3, "new_user", "stacking_train"),
        ],
        [
            "user_id",
            "order_number",
            "order_history",
            "user_cohort",
            "development_split",
        ],
    )

    result = select_stacking_model_users(df)

    actual = {
        (row["user_id"], row["order_number"]): row["order_role"]
        for row in result.collect()
    }

    assert actual == {
        (1, 1): "history",
        (1, 2): "history",
        (1, 3): "train_label",
        (1, 4): "validation_label",
    }
