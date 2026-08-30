import pytest
from pyspark.sql import functions as F

from instacart_etl_rnn.simulation.create_user_split import build_user_simulation_split


def test_build_user_simulation_split_calculates_order_history(spark):
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
            (2, 6),
        ],
        ["user_id", "order_number"],
    )

    result = build_user_simulation_split(orders)

    actual = {row.user_id: row.order_history for row in result.collect()}

    assert actual == {
        1: 3,
        2: 6,
    }


def test_build_user_simulation_split_excludes_users_with_less_than_three_orders(
    spark,
):
    orders = spark.createDataFrame(
        [
            (1, 1),
            (1, 2),
        ],
        ["user_id", "order_number"],
    )

    result = build_user_simulation_split(orders)

    row = result.first()

    assert row.user_cohort == "excluded"
    assert row.development_split is None
    assert row.arrival_period is None


@pytest.mark.parametrize(
    "order_history",
    [3, 4, 5],
)
def test_build_user_simulation_split_assigns_new_user_cohort(
    spark,
    order_history,
):
    orders = spark.createDataFrame(
        [(1, order_number) for order_number in range(1, order_history + 1)],
        ["user_id", "order_number"],
    )

    result = build_user_simulation_split(orders)

    row = result.first()

    assert row.user_cohort == "new_user"
    assert row.development_split is None
    assert row.arrival_period in {"t1", "t2"}


def test_build_user_simulation_split_established_users_have_development_split(
    spark,
):
    orders = spark.createDataFrame(
        [
            (user_id, order_number)
            for user_id in range(1, 101)
            for order_number in range(1, 7)
        ],
        ["user_id", "order_number"],
    )

    result = build_user_simulation_split(orders)

    established = result.filter(F.col("user_cohort") == "established").collect()

    assert established

    for row in established:
        assert row.development_split in {
            "base_train",
            "stacking_train",
        }
        assert row.arrival_period is None


def test_build_user_simulation_split_final_holdout_has_no_other_split(
    spark,
):
    orders = spark.createDataFrame(
        [
            (user_id, order_number)
            for user_id in range(1, 101)
            for order_number in range(1, 7)
        ],
        ["user_id", "order_number"],
    )

    result = build_user_simulation_split(orders)

    holdout = result.filter(F.col("user_cohort") == "final_holdout").collect()

    assert holdout

    for row in holdout:
        assert row.development_split is None
        assert row.arrival_period is None


def test_build_user_simulation_split_new_users_receive_arrival_period(
    spark,
):
    orders = spark.createDataFrame(
        [
            (user_id, order_number)
            for user_id in range(1, 21)
            for order_number in range(1, 5)
        ],
        ["user_id", "order_number"],
    )

    result = build_user_simulation_split(orders)

    rows = result.collect()

    assert len(rows) == 20

    for row in rows:
        assert row.user_cohort == "new_user"
        assert row.development_split is None
        assert row.arrival_period in {"t1", "t2"}


def test_build_user_simulation_split_returns_one_row_per_user(spark):
    orders = spark.createDataFrame(
        [
            (1, 1),
            (1, 2),
            (1, 3),
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
        ],
        ["user_id", "order_number"],
    )

    result = build_user_simulation_split(orders)

    assert result.count() == 2

    assert {row.user_id for row in result.select("user_id").collect()} == {1, 2}


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
