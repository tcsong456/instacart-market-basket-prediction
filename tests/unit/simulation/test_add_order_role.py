import pytest

from instacart_etl_rnn.simulation.create_user_split import add_order_role


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        (
            "initial",
            {
                1: (True, True, False),
                2: (True, True, False),
                3: (True, True, False),
                4: (False, True, False),
                5: (False, False, False),
                6: (False, False, False),
            },
        ),
        (
            "t1",
            {
                1: (True, True, False),
                2: (True, True, False),
                3: (True, True, False),
                4: (True, True, False),
                5: (False, True, False),
                6: (False, False, False),
            },
        ),
        (
            "t2",
            {
                1: (True, True, False),
                2: (True, True, False),
                3: (True, True, False),
                4: (True, True, False),
                5: (True, True, False),
                6: (False, True, False),
            },
        ),
    ],
)
def test_add_order_role_for_established_user(
    spark,
    period,
    expected,
):
    df = spark.createDataFrame(
        [(1, order_number, 6, "established", None) for order_number in range(1, 7)],
        """
        user_id int,
        order_number int,
        order_history int,
        user_cohort string,
        arrival_period string
        """,
    )

    result = add_order_role(df, period)

    actual = {
        row.order_number: (
            row.is_train_available,
            row.is_validation_available,
            row.is_evaluation_available,
        )
        for row in result.collect()
    }

    assert actual == expected


def test_add_order_role_established_user_progresses_across_periods(spark):
    df = spark.createDataFrame(
        [(101, order_number, 6, "established", None) for order_number in range(1, 7)],
        """
        user_id int,
        order_number int,
        order_history int,
        user_cohort string,
        arrival_period string
        """,
    )

    expected_by_period = {
        "initial": {
            1: (True, True, False),
            2: (True, True, False),
            3: (True, True, False),
            4: (False, True, False),
            5: (False, False, False),
            6: (False, False, False),
        },
        "t1": {
            1: (True, True, False),
            2: (True, True, False),
            3: (True, True, False),
            4: (True, True, False),
            5: (False, True, False),
            6: (False, False, False),
        },
        "t2": {
            1: (True, True, False),
            2: (True, True, False),
            3: (True, True, False),
            4: (True, True, False),
            5: (True, True, False),
            6: (False, True, False),
        },
    }

    flags_by_period = {}

    for period, expected in expected_by_period.items():
        result = add_order_role(df, period)

        actual = {
            row.order_number: (
                row.is_train_available,
                row.is_validation_available,
                row.is_evaluation_available,
            )
            for row in result.collect()
        }

        assert {row.current_period for row in result.collect()} == {period}
        assert actual == expected
        flags_by_period[period] = actual

    train_initial = {
        order_number
        for order_number, flags in flags_by_period["initial"].items()
        if flags[0]
    }
    train_t1 = {
        order_number
        for order_number, flags in flags_by_period["t1"].items()
        if flags[0]
    }
    train_t2 = {
        order_number
        for order_number, flags in flags_by_period["t2"].items()
        if flags[0]
    }

    validation_initial = {
        order_number
        for order_number, flags in flags_by_period["initial"].items()
        if flags[1]
    }
    validation_t1 = {
        order_number
        for order_number, flags in flags_by_period["t1"].items()
        if flags[1]
    }
    validation_t2 = {
        order_number
        for order_number, flags in flags_by_period["t2"].items()
        if flags[1]
    }

    assert train_initial < train_t1 < train_t2
    assert validation_initial < validation_t1 < validation_t2


@pytest.mark.parametrize(
    ("arrival_period", "period"),
    [
        ("t1", "t1"),
        ("t2", "t2"),
    ],
)
def test_add_order_role_current_new_user(
    spark,
    arrival_period,
    period,
):
    df = spark.createDataFrame(
        [
            (1, order_number, 4, "new_user", arrival_period)
            for order_number in range(1, 5)
        ],
        """
        user_id int,
        order_number int,
        order_history int,
        user_cohort string,
        arrival_period string
        """,
    )

    result = add_order_role(df, period)

    actual = {
        row.order_number: (
            row.is_train_available,
            row.is_validation_available,
            row.is_evaluation_available,
        )
        for row in result.collect()
    }

    assert actual == {
        1: (True, True, False),
        2: (True, True, False),
        3: (True, True, False),
        4: (False, True, False),
    }


def test_add_order_role_previous_new_user_is_train_available_only(
    spark,
):
    df = spark.createDataFrame(
        [(1, order_number, 4, "new_user", "t1") for order_number in range(1, 5)],
        """
        user_id int,
        order_number int,
        order_history int,
        user_cohort string,
        arrival_period string
        """,
    )

    result = add_order_role(df, "t2")

    for row in result.collect():
        assert row.is_train_available is True
        assert row.is_validation_available is False
        assert row.is_evaluation_available is False


@pytest.mark.parametrize(
    ("period", "available_orders"),
    [
        ("initial", {1, 2, 3, 4}),
        ("t1", {1, 2, 3, 4, 5}),
        ("t2", {1, 2, 3, 4, 5, 6}),
    ],
)
def test_add_order_role_final_holdout_evaluation_availability(
    spark,
    period,
    available_orders,
):
    df = spark.createDataFrame(
        [(1, order_number, 6, "final_holdout", None) for order_number in range(1, 7)],
        """
        user_id int,
        order_number int,
        order_history int,
        user_cohort string,
        arrival_period string
        """,
    )

    result = add_order_role(df, period)

    actual = {
        row.order_number for row in result.collect() if row.is_evaluation_available
    }

    assert actual == available_orders

    for row in result.collect():
        assert row.is_train_available is False
        assert row.is_validation_available is False


def test_add_order_role_evaluation_available_only_for_final_holdout(spark):
    df = spark.createDataFrame(
        [
            *[(101, n, 6, "established", None) for n in range(1, 7)],
            *[(202, n, 4, "new_user", "t2") for n in range(1, 5)],
            *[(303, n, 6, "final_holdout", None) for n in range(1, 7)],
            *[(404, n, 2, "excluded", None) for n in range(1, 3)],
        ],
        """
        user_id int,
        order_number int,
        order_history int,
        user_cohort string,
        arrival_period string
        """,
    )

    result = add_order_role(df, "t2")
    rows = result.collect()

    evaluation_rows = [row for row in rows if row.is_evaluation_available]

    assert evaluation_rows
    assert {row.user_id for row in evaluation_rows} == {303}
    assert {row.user_cohort for row in evaluation_rows} == {"final_holdout"}
    assert {row.order_number for row in evaluation_rows} == {1, 2, 3, 4, 5, 6}

    for row in rows:
        if row.user_cohort != "final_holdout":
            assert row.is_evaluation_available is False


def test_add_order_role_excluded_user_has_no_available_orders(
    spark,
):
    df = spark.createDataFrame(
        [
            (1, 1, 2, "excluded", None),
            (1, 2, 2, "excluded", None),
        ],
        """
        user_id int,
        order_number int,
        order_history int,
        user_cohort string,
        arrival_period string
        """,
    )

    result = add_order_role(df, "initial")

    for row in result.collect():
        assert row.is_train_available is False
        assert row.is_validation_available is False
        assert row.is_evaluation_available is False


@pytest.mark.parametrize(
    "period",
    ["initial", "t1", "t2"],
)
def test_add_order_role_sets_current_period(
    spark,
    period,
):
    df = spark.createDataFrame(
        [(1, 1, 6, "established", None)],
        """
        user_id int,
        order_number int,
        order_history int,
        user_cohort string,
        arrival_period string
        """,
    )

    result = add_order_role(df, period)

    assert result.first().current_period == period


def test_add_order_role_rejects_unsupported_period(spark):
    df = spark.createDataFrame(
        [(1, 1, 6, "established", None)],
        """
        user_id int,
        order_number int,
        order_history int,
        user_cohort string,
        arrival_period string
        """,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported simulation period: t3",
    ):
        add_order_role(df, "t3")
