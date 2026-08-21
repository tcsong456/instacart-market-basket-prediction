from instacart_etl_rnn.gold.create_aisle_training_data import (
    PARSE_COLUMNS,
    parse_aisle_seq_data,
)


def test_parse_aisle_seq_data_preserves_extra_temporal_step(spark):
    df = spark.createDataFrame(
        [
            (
                1,
                5,
                50,
                "train",
                "1 0 1",
                "2 0 3",
                "1 2 1",
                "3 4 5",
                "0 1 2 3",
                "10 11 12 13",
                "-1 7 5 4",
                "1 2 3 4",
            ),
        ],
        """
        user_id int,
        aisle_id int,
        department_id int,
        eval_set string,
        is_ordered_history string,
        position_in_order string,
        num_products_from_aisle string,
        aisle_history_size string,
        order_dows string,
        order_hours string,
        days_since_prior_orders string,
        order_numbers string,
        """,
    )

    result = parse_aisle_seq_data(
        df,
        max_padded_length=5,
    )

    row = result.first()

    assert row["user_id"] == 1
    assert row["aisle_id"] == 5
    assert row["department_id"] == 50
    assert row["eval_set"] == "train"

    assert row["is_ordered_history"] == [1, 0, 1, 0, 0]
    assert row["position_in_order"] == [2, 0, 3, 0, 0]
    assert row["num_products_from_aisle"] == [1, 2, 1, 0, 0]
    assert row["aisle_history_size"] == [3, 4, 5, 0, 0]

    assert row["history_length"] == 3

    assert row["order_dows"] == [0, 1, 2, 3, 0]
    assert row["order_hours"] == [10, 11, 12, 13, 0]
    assert row["days_since_prior_orders"] == [-1.0, 7.0, 5.0, 4.0, 0.0]
    assert row["order_numbers"] == [1, 2, 3, 4, 0]

    assert set(result.columns) == {
        "user_id",
        "aisle_id",
        "department_id",
        "eval_set",
        "is_ordered_history",
        "position_in_order",
        "num_products_from_aisle",
        "aisle_history_size",
        "order_dows",
        "order_hours",
        "days_since_prior_orders",
        "order_numbers",
        "history_length",
    }


def test_parse_aisle_seq_data_truncates_both_sequence_groups(spark):
    df = spark.createDataFrame(
        [
            (
                1,
                5,
                50,
                "train",
                "1 1 0 1 1",
                "1 2 0 3 4",
                "2 2 1 3 4",
                "5 6 4 7 8",
                "0 1 2 3 4 5",
                "10 11 12 13 14 15",
                "-1 7 5 4 3 2",
                "1 2 3 4 5 6",
            ),
        ],
        """
        user_id int,
        aisle_id int,
        department_id int,
        eval_set string,
        is_ordered_history string,
        position_in_order string,
        num_products_from_aisle string,
        aisle_history_size string,
        order_dows string,
        order_hours string,
        days_since_prior_orders string,
        order_numbers string,
        """,
    )

    result = parse_aisle_seq_data(
        df,
        max_padded_length=4,
    )

    row = result.first()

    assert row["is_ordered_history"] == [1, 1, 0, 1]
    assert row["history_length"] == 4

    assert row["position_in_order"] == [1, 2, 0, 3]
    assert row["num_products_from_aisle"] == [2, 2, 1, 3]
    assert row["aisle_history_size"] == [5, 6, 4, 7]

    assert row["order_dows"] == [0, 1, 2, 3]
    assert row["order_hours"] == [10, 11, 12, 13]
    assert row["days_since_prior_orders"] == [-1.0, 7.0, 5.0, 4.0]
    assert row["order_numbers"] == [1, 2, 3, 4]


def test_parse_aisle_seq_data_handles_exact_history_boundary(spark):
    df = spark.createDataFrame(
        [
            (
                1,
                5,
                50,
                "train",
                "1 0 1 1",
                "2 0 3 1",
                "1 2 1 3",
                "3 4 5 6",
                "0 1 2 3 4",
                "10 11 12 13 14",
                "-1 7 5 4 3",
                "1 2 3 4 5",
            ),
        ],
        """
        user_id int,
        aisle_id int,
        department_id int,
        eval_set string,
        is_ordered_history string,
        position_in_order string,
        num_products_from_aisle string,
        aisle_history_size string,
        order_dows string,
        order_hours string,
        days_since_prior_orders string,
        order_numbers string,
        """,
    )

    result = parse_aisle_seq_data(
        df,
        max_padded_length=4,
    )

    row = result.first()

    assert row["is_ordered_history"] == [1, 0, 1, 1]
    assert row["position_in_order"] == [2, 0, 3, 1]
    assert row["num_products_from_aisle"] == [1, 2, 1, 3]
    assert row["aisle_history_size"] == [3, 4, 5, 6]

    assert row["history_length"] == 4

    assert row["order_dows"] == [0, 1, 2, 3]
    assert row["order_hours"] == [10, 11, 12, 13]
    assert row["days_since_prior_orders"] == [-1.0, 7.0, 5.0, 4.0]
    assert row["order_numbers"] == [1, 2, 3, 4]


def test_parse_aisle_seq_data_handles_empty_sequences(spark):
    df = spark.createDataFrame(
        [
            (
                1,
                5,
                50,
                "train",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
            ),
        ],
        """
        user_id int,
        aisle_id int,
        department_id int,
        eval_set string,
        is_ordered_history string,
        position_in_order string,
        num_products_from_aisle string,
        aisle_history_size string,
        order_dows string,
        order_hours string,
        days_since_prior_orders string,
        order_numbers string
        """,
    )

    result = parse_aisle_seq_data(
        df,
        max_padded_length=3,
    )

    row = result.first()

    assert row["history_length"] == 0

    for colname in PARSE_COLUMNS:
        assert row[colname] == [0, 0, 0]
