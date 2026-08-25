from instacart_etl_rnn.simulation.create_user_split import build_user_simulation_split


def test_build_user_simulation_split_assigns_user_cohorts(
    spark,
):
    orders = spark.createDataFrame(
        [
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (2, 3),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (3, 6),
        ],
        ["user_id", "order_number"],
    )

    result = build_user_simulation_split(orders)

    actual = {
        row.user_id: (
            row.order_history,
            row.user_cohort,
        )
        for row in result.collect()
    }

    assert actual[1][0] == 2
    assert actual[2][0] == 3
    assert actual[3][0] == 6

    assert actual[1][1] in {"excluded", "final_holdout"}
    assert actual[2][1] in {"new_user", "final_holdout"}
    assert actual[3][1] in {"established", "final_holdout"}


def test_build_user_simulation_split_only_established_users_have_development_split(
    spark,
):
    orders = spark.createDataFrame(
        [
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
            (2, 3),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (3, 6),
        ],
        ["user_id", "order_number"],
    )

    result = build_user_simulation_split(orders)

    rows = {row.user_id: row for row in result.collect()}

    for row in rows.values():
        if row.user_cohort == "established":
            assert row.development_split in {
                "base_train",
                "stacking_train",
            }
        else:
            assert row.development_split is None


def test_build_user_simulation_split_only_new_users_have_arrival_period(
    spark,
):
    orders = spark.createDataFrame(
        [
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (3, 5),
            (3, 6),
        ],
        ["user_id", "order_number"],
    )

    result = build_user_simulation_split(orders)

    for row in result.collect():
        if row.user_cohort == "new_user":
            assert row.arrival_period in {
                "t1",
                "t2",
            }
        else:
            assert row.arrival_period is None


def test_build_user_simulation_split_assigns_deterministic_arrival_period(
    spark,
):
    orders = spark.createDataFrame(
        [
            (10, 1),
            (10, 2),
            (10, 3),
            (11, 1),
            (11, 2),
            (11, 3),
            (12, 1),
            (12, 2),
            (12, 3),
        ],
        ["user_id", "order_number"],
    )

    result_1 = build_user_simulation_split(orders)
    result_2 = build_user_simulation_split(orders)

    actual_1 = {row.user_id: row.arrival_period for row in result_1.collect()}

    actual_2 = {row.user_id: row.arrival_period for row in result_2.collect()}

    assert actual_1 == actual_2
