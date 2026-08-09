from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from instacart_etl_rnn.silver.create_user_data import build_order_level_data


def test_build_order_level_data_groups_and_orders_items(spark):
    df = spark.createDataFrame(
        [
            (1, 100, 2, 20, 1, 5, 50, 3, 1, 14, 7.0, "prior"),
            (1, 100, 1, 10, 0, 4, 40, 3, 1, 14, 7.0, "prior"),
            (1, 100, 3, 30, 1, 6, 60, 3, 1, 14, 7.0, "prior"),
            (1, 101, 1, 99, 0, 9, 90, 4, 2, 16, 5.0, "train"),
        ],
        [
            "user_id",
            "order_id",
            "add_to_cart_order",
            "product_id",
            "reordered",
            "aisle_id",
            "department_id",
            "order_number",
            "order_dow",
            "order_hour_of_day",
            "days_since_prior_order",
            "eval_set",
        ],
    )

    result = build_order_level_data(df)

    rows = {row["order_id"]: row for row in result.collect()}

    order_100 = rows[100]

    assert order_100["user_id"] == 1
    assert order_100["products"] == "10_20_30"
    assert order_100["reorders"] == "0_1_1"
    assert order_100["aisles"] == "4_5_6"
    assert order_100["departments"] == "40_50_60"

    assert order_100["order_number"] == 3
    assert order_100["order_dow"] == 1
    assert order_100["order_hour"] == 14
    assert order_100["days_since_prior_order"] == 7.0
    assert order_100["eval_set"] == "prior"

    order_101 = rows[101]

    assert order_101["products"] == "99"
    assert order_101["reorders"] == "0"
    assert order_101["aisles"] == "9"
    assert order_101["departments"] == "90"
    assert order_101["eval_set"] == "train"

    assert "items" not in result.columns


def test_build_order_level_data_handles_empty_dataframe(spark):
    schema = StructType(
        [
            StructField("product_id", IntegerType(), False),
            StructField("order_id", IntegerType(), False),
            StructField("add_to_cart_order", IntegerType(), False),
            StructField("reordered", IntegerType(), False),
            StructField("user_id", IntegerType(), False),
            StructField("eval_set", StringType(), False),
            StructField("order_number", IntegerType(), False),
            StructField("order_dow", IntegerType(), False),
            StructField("order_hour_of_day", IntegerType(), False),
            StructField("days_since_prior_order", DoubleType(), False),
            StructField("product_name", StringType(), False),
            StructField("aisle_id", IntegerType(), False),
            StructField("department_id", IntegerType(), False),
        ]
    )

    df = spark.createDataFrame(
        [],
        schema=schema,
    )

    result = build_order_level_data(df)

    assert result.isEmpty()

    assert result.columns == [
        "user_id",
        "order_id",
        "order_number",
        "order_dow",
        "order_hour",
        "days_since_prior_order",
        "eval_set",
        "products",
        "reorders",
        "aisles",
        "departments",
    ]
