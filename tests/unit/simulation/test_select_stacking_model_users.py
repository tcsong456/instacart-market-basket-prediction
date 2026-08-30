from instacart_etl_rnn.simulation.create_order_product_split import (
    select_stacking_model_users,
)


def test_select_stacking_model_users_filters_and_sets_availability(
    spark,
):
    df = spark.createDataFrame(
        [
            (1, 1, 4, "established", "stacking_train", False, False),
            (1, 2, 4, "established", "stacking_train", False, False),
            (1, 3, 4, "established", "stacking_train", False, False),
            (1, 4, 4, "established", "stacking_train", False, False),
            (2, 1, 4, "established", "base_train", True, True),
            (3, 1, 4, "new_user", None, True, True),
            (4, 1, 4, "final_holdout", None, False, False),
        ],
        """
        user_id int,
        order_number int,
        order_history int,
        user_cohort string,
        development_split string,
        is_train_available boolean,
        is_validation_available boolean
        """,
    )

    result = select_stacking_model_users(df)

    actual = {
        row.order_number: (
            row.is_train_available,
            row.is_validation_available,
        )
        for row in result.collect()
    }

    assert actual == {
        1: (True, True),
        2: (True, True),
        3: (True, True),
        4: (False, True),
    }
