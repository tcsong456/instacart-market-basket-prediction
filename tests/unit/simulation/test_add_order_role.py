import pytest

from instacart_etl_rnn.simulation.create_user_split import add_order_role


@pytest.mark.parametrize(
    ("period", "expected_roles"),
    [
        (
            "initial",
            {
                1: "history",
                2: "history",
                3: "train_label",
                4: "validation_label",
                5: "future",
                6: "future",
            },
        ),
        (
            "t1",
            {
                1: "history",
                2: "history",
                3: "history",
                4: "train_label",
                5: "validation_label",
                6: "future",
            },
        ),
        (
            "t2",
            {
                1: "history",
                2: "history",
                3: "history",
                4: "history",
                5: "train_label",
                6: "validation_label",
            },
        ),
    ],
)
def test_add_order_role_assigns_established_user_roles(
    spark,
    period,
    expected_roles,
):
    df = spark.createDataFrame(
        [
            (1, 1, 6, "established", None),
            (1, 2, 6, "established", None),
            (1, 3, 6, "established", None),
            (1, 4, 6, "established", None),
            (1, 5, 6, "established", None),
            (1, 6, 6, "established", None),
        ],
        [
            "user_id",
            "order_number",
            "order_history",
            "user_cohort",
            "arrival_period",
        ],
    )

    result = add_order_role(df, period)

    actual = {
        row["order_number"]: row["order_role"]
        for row in result.select(
            "order_number",
            "order_role",
        ).collect()
    }

    assert actual == expected_roles

    assert {
        row["current_period"]
        for row in result.select("current_period").distinct().collect()
    } == {period}


def test_add_order_role_marks_new_user_before_arrival_as_future(
    spark,
):
    df = spark.createDataFrame(
        [
            (10, 1, 4, "new_user", "t2"),
            (10, 2, 4, "new_user", "t2"),
            (10, 3, 4, "new_user", "t2"),
            (10, 4, 4, "new_user", "t2"),
        ],
        [
            "user_id",
            "order_number",
            "order_history",
            "user_cohort",
            "arrival_period",
        ],
    )

    result = add_order_role(df, period="t1")

    actual = {row["order_number"]: row["order_role"] for row in result.collect()}

    assert actual == {
        1: "future",
        2: "future",
        3: "future",
        4: "future",
    }


def test_add_order_role_assigns_current_new_user_roles(
    spark,
):
    df = spark.createDataFrame(
        [
            (10, 1, 4, "new_user", "t1"),
            (10, 2, 4, "new_user", "t1"),
            (10, 3, 4, "new_user", "t1"),
            (10, 4, 4, "new_user", "t1"),
        ],
        [
            "user_id",
            "order_number",
            "order_history",
            "user_cohort",
            "arrival_period",
        ],
    )

    result = add_order_role(df, period="t1")

    actual = {row["order_number"]: row["order_role"] for row in result.collect()}

    assert actual == {
        1: "history",
        2: "history",
        3: "train_label",
        4: "validation_label",
    }


def test_add_order_role_moves_previous_new_user_validation_into_train(
    spark,
):
    df = spark.createDataFrame(
        [
            (10, 1, 4, "new_user", "t1"),
            (10, 2, 4, "new_user", "t1"),
            (10, 3, 4, "new_user", "t1"),
            (10, 4, 4, "new_user", "t1"),
        ],
        [
            "user_id",
            "order_number",
            "order_history",
            "user_cohort",
            "arrival_period",
        ],
    )

    result = add_order_role(df, period="t2")

    actual = {row["order_number"]: row["order_role"] for row in result.collect()}

    assert actual == {
        1: "history",
        2: "history",
        3: "history",
        4: "train_label",
    }


def test_add_order_role_rejects_unsupported_period(spark):
    df = spark.createDataFrame(
        [],
        """
        user_id long,
        order_number long,
        order_history long,
        user_cohort string,
        arrival_period string
        """,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported simulation period: t3",
    ):
        add_order_role(df, period="t3")
