from instacart_etl_rnn.simulation.create_order_product_split import (
    select_base_model_users,
)


def test_select_base_model_users(spark):
    df = spark.createDataFrame(
        [
            (1, "established", "base_train"),
            (2, "established", "stacking_train"),
            (3, "new_user", None),
            (4, "final_holdout", None),
            (5, "excluded", None),
        ],
        """
        user_id int,
        user_cohort string,
        development_split string
        """,
    )

    result = select_base_model_users(df)

    actual = {row.user_id for row in result.collect()}

    assert actual == {1, 3, 4}
