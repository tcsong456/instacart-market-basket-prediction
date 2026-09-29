import pytest

from instacart_etl_rnn.simulation.create_order_product_split import (
    select_stacking_model_users,
    split_order_products_by_role,
)


@pytest.mark.parametrize(
    ("period", "train_orders"),
    [
        ("initial", {1, 2}),
        ("t1", {1, 2, 3}),
        ("t2", {1, 2, 3, 4}),
    ],
)
def test_select_stacking_model_users_filters_and_sets_availability(
    spark,
    period,
    train_orders,
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

    result = select_stacking_model_users(df, period)

    actual = {
        row.order_number: (
            row.is_train_available,
            row.is_validation_available,
        )
        for row in result.collect()
    }

    assert actual == {
        order_number: (order_number in train_orders, True)
        for order_number in range(1, 5)
    }


@pytest.mark.parametrize(
    ("period", "train_orders"),
    [
        ("initial", {1, 2, 3, 4}),
        ("t1", {1, 2, 3, 4, 5}),
        ("t2", {1, 2, 3, 4, 5, 6}),
    ],
)
def test_select_stacking_model_users_rewrites_train_val_leaves_evaluation_untouched(
    spark,
    period,
    train_orders,
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

    result = select_stacking_model_users(df, period)
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
        order_number: (order_number in train_orders, True, False)
        for order_number in range(1, 7)
    }
    assert all(row.is_evaluation_available is False for row in rows)

    train_history, evaluation_history, validation_history = (
        split_order_products_by_role(result)
    )

    assert {row.order_number for row in train_history.collect()} == train_orders
    assert {row.order_number for row in validation_history.collect()} == {
        1,
        2,
        3,
        4,
        5,
        6,
    }
    assert evaluation_history.count() == 0


def test_select_stacking_model_users_rejects_unsupported_period(spark):
    df = spark.createDataFrame(
        [
            (1, 1, 6, "established", "stacking_train", False, False),
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

    with pytest.raises(ValueError, match="Unsupported simulation period: t3"):
        select_stacking_model_users(df, "t3")
