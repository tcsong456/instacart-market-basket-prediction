from instacart_etl_rnn.simulation.create_order_product_split import (
    select_stacking_model_users,
    split_order_products_by_role,
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


def test_select_stacking_model_users_rewrites_train_val_leaves_evaluation_untouched(
    spark,
):
    """Rewrite train/val for stacking users; evaluation stays false (established)."""

    df = spark.createDataFrame(
        [
            (1, 1, 6, "established", "stacking_train", True, True, False),
            (1, 2, 6, "established", "stacking_train", True, True, False),
            (1, 3, 6, "established", "stacking_train", True, True, False),
            (1, 4, 6, "established", "stacking_train", False, True, False),
            (1, 5, 6, "established", "stacking_train", False, False, False),
            (1, 6, 6, "established", "stacking_train", False, False, False),
            (2, 1, 6, "established", "base_train", True, True, False),
            (3, 1, 4, "new_user", None, True, True, False),
            (4, 1, 6, "final_holdout", None, False, False, True),
        ],
        """
        user_id int,
        order_number int,
        order_history int,
        user_cohort string,
        development_split string,
        is_train_available boolean,
        is_validation_available boolean,
        is_evaluation_available boolean
        """,
    )

    result = select_stacking_model_users(df)
    rows = result.orderBy("order_number").collect()

    assert [row.user_id for row in rows] == [1, 1, 1, 1, 1, 1]

    actual = {
        row.order_number: (
            row.is_train_available,
            row.is_validation_available,
            row.is_evaluation_available,
        )
        for row in rows
    }

    assert actual == {
        1: (True, True, False),
        2: (True, True, False),
        3: (True, True, False),
        4: (True, True, False),
        5: (True, True, False),
        6: (False, True, False),
    }
    assert all(row.is_evaluation_available is False for row in rows)

    train_history, evaluation_history, validation_history = (
        split_order_products_by_role(result)
    )

    assert {row.order_number for row in train_history.collect()} == {1, 2, 3, 4, 5}
    assert {row.order_number for row in validation_history.collect()} == {
        1,
        2,
        3,
        4,
        5,
        6,
    }
    assert evaluation_history.count() == 0
