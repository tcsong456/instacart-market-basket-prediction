from pyspark.sql.types import IntegerType

from instacart_etl_rnn.gold.create_user_product_count_data import (
    build_user_product_count,
)


def test_build_user_product_count_counts_prior_purchases(spark):
    df = spark.createDataFrame(
        [
            (1, 10, "prior"),
            (1, 10, "prior"),
            (1, 20, "prior"),
            (1, 10, "train"),
            (2, 10, "prior"),
            (2, 10, "prior"),
            (2, 30, "test"),
        ],
        ["user_id", "product_id", "eval_set"],
    )

    result = build_user_product_count(df)

    actual = {
        (row["user_id"], row["product_id"]): row["count"] for row in result.collect()
    }

    assert actual == {
        (1, 10): 2,
        (1, 20): 1,
        (2, 10): 2,
    }

    assert result.columns == [
        "user_id",
        "product_id",
        "count",
    ]

    assert isinstance(
        result.schema["count"].dataType,
        IntegerType,
    )
