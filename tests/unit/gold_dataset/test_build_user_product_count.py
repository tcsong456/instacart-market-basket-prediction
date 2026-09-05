from pyspark.sql.types import IntegerType

from instacart_etl_rnn.gold.create_user_product_count_data import (
    build_user_product_count,
)


def test_build_user_product_count_excludes_each_users_latest_order(spark):
    order_products = spark.createDataFrame(
        [
            (1, 101, 1, 10),
            (1, 101, 1, 10),
            (1, 102, 2, 10),
            (1, 102, 2, 20),
            (1, 103, 3, 10),
            (1, 103, 3, 30),
            (2, 201, 1, 40),
            (2, 202, 2, 40),
        ],
        ["user_id", "order_id", "order_number", "product_id"],
    )

    result = build_user_product_count(order_products)

    actual = {(row.user_id, row.product_id): row["count"] for row in result.collect()}

    assert actual == {
        (1, 10): 2,
        (1, 20): 1,
        (2, 40): 1,
    }
    assert result.columns == ["user_id", "product_id", "count"]
    assert isinstance(result.schema["count"].dataType, IntegerType)


def test_build_user_product_count_returns_no_rows_without_history(spark):
    order_products = spark.createDataFrame(
        [(1, 101, 1, 10)],
        ["user_id", "order_id", "order_number", "product_id"],
    )

    result = build_user_product_count(order_products)

    assert result.count() == 0
