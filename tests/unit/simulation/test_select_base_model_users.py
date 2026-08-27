from instacart_etl_rnn.simulation.create_order_product_split import (
    select_base_model_users,
)


def test_select_base_model_users_returns_expected_rows(spark):
    df = spark.createDataFrame(
        [
            (1, "established", "base_train", 10),
            (2, "established", "stacking_train", 20),
            (3, "new_user", None, 30),
        ],
        [
            "user_id",
            "user_cohort",
            "development_split",
            "order_number",
        ],
    )

    result = select_base_model_users(df)

    actual = {
        (
            row["user_id"],
            row["user_cohort"],
            row["development_split"],
            row["order_number"],
        )
        for row in result.collect()
    }

    assert actual == {
        (1, "established", "base_train", 10),
        (3, "new_user", None, 30),
    }
