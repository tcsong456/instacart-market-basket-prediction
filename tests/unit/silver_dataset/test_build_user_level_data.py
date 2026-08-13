from instacart_etl_rnn.silver.create_user_data import build_user_level_data


def test_build_user_level_data_builds_order_sequences(spark):
    df = spark.createDataFrame(
        [
            (
                1,
                102,
                2,
                3,
                15,
                7.0,
                "20_21",
                "1_0",
                "5_6",
                "2_3",
                "prior",
            ),
            (
                1,
                101,
                1,
                2,
                10,
                -1.0,
                "10_11_12",
                "0_0_1",
                "4_2_5",
                "3_5_20",
                "prior",
            ),
            (
                1,
                103,
                3,
                4,
                18,
                5.0,
                "30",
                "1",
                "7",
                "4",
                "train",
            ),
            (
                2,
                201,
                1,
                1,
                9,
                -1.0,
                "50_150_80",
                "0_1_0",
                "8_15_11",
                "5_1_17",
                "prior",
            ),
        ],
        [
            "user_id",
            "order_id",
            "order_number",
            "order_dow",
            "order_hour",
            "days_since_prior_order",
            "products",
            "reorders",
            "aisles",
            "departments",
            "eval_set",
        ],
    )

    result = build_user_level_data(df)

    rows = {row["user_id"]: row for row in result.collect()}

    user_1 = rows[1]

    assert user_1["order_ids"] == "101 102 103"
    assert user_1["order_numbers"] == "1 2 3"
    assert user_1["order_dows"] == "2 3 4"
    assert user_1["order_hours"] == "10 15 18"

    assert user_1["days_since_prior_orders"] == ("-1.0 7.0 5.0")

    assert user_1["product_ids"] == ("10_11_12 20_21 30")

    assert user_1["reorders"] == ("0_0_1 1_0 1")

    assert user_1["aisle_ids"] == ("4_2_5 5_6 7")

    assert user_1["department_ids"] == ("3_5_20 2_3 4")

    assert user_1["eval_set"] == "train"

    user_2 = rows[2]

    assert user_2["order_ids"] == "201"
    assert user_2["order_numbers"] == "1"
    assert user_2["product_ids"] == "50_150_80"
    assert user_2["eval_set"] == "test"

    assert "orders" not in result.columns


def test_build_user_level_data_handles_empty_dataframe(
    spark,
):
    schema = """
        user_id INT,
        order_id INT,
        order_number INT,
        order_dow INT,
        order_hour INT,
        days_since_prior_order DOUBLE,
        products STRING,
        reorders STRING,
        aisles STRING,
        departments STRING,
        eval_set STRING
    """

    df = spark.createDataFrame(
        [],
        schema=schema,
    )

    result = build_user_level_data(df)

    assert result.isEmpty()

    assert result.columns == [
        "user_id",
        "order_ids",
        "order_numbers",
        "order_dows",
        "order_hours",
        "days_since_prior_orders",
        "product_ids",
        "reorders",
        "department_ids",
        "aisle_ids",
        "eval_set",
    ]
