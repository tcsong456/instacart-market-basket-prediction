from instacart_etl_rnn.simulation.create_user_split import build_order_simulation_split


def test_build_order_simulation_split_assigns_expected_periods(spark):
    orders = spark.createDataFrame(
        [
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (2, 1),
            (2, 2),
            (3, 1),
            (4, 1),
        ],
        ["user_id", "order_number"],
    )

    user_split = spark.createDataFrame(
        [
            (1, "established", 6),
            (2, "new_user", 2),
            (3, "final_holdout", 1),
            (4, "excluded", 1),
        ],
        ["user_id", "user_cohort", "order_history"],
    )

    result = build_order_simulation_split(
        orders=orders,
        user_split=user_split,
    )

    actual = {
        (row["user_id"], row["order_number"]): row["simulation_period"]
        for row in result.collect()
    }

    expected = {
        (1, 1): "initial",
        (1, 2): "initial",
        (1, 3): "initial",
        (1, 4): "validation",
        (1, 5): "t1",
        (1, 6): "t2",
        (2, 1): "new_user_pool",
        (2, 2): "new_user_pool",
        (3, 1): "final_holdout",
        (4, 1): "excluded",
    }

    assert actual == expected


def test_build_order_simulation_split_drops_users_missing_from_user_split(
    spark,
):
    orders = spark.createDataFrame(
        [
            (1, 1),
            (999, 1),
        ],
        ["user_id", "order_number"],
    )

    user_split = spark.createDataFrame(
        [
            (1, "final_holdout", 1),
        ],
        ["user_id", "user_cohort", "order_history"],
    )

    result = build_order_simulation_split(
        orders,
        user_split,
    )

    actual_user_ids = {row["user_id"] for row in result.select("user_id").collect()}

    assert actual_user_ids == {1}
