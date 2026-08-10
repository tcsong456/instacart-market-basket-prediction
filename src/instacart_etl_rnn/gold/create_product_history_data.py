from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from instacart_etl_rnn.common.io import read_parquet
from instacart_etl_rnn.common.paths import join_path

SELECTED_COLUMNS = [
    "user_id",
    "product_id",
    "label",
    "aisle_id",
    "department_id",
    "product_name",
    "eval_set",
    "is_ordered_history",
    "position_in_order_history",
    "history_order_size",
    "history_reorder_size",
    "order_dows",
    "order_hours",
    "days_since_prior_orders",
    "order_numbers",
]


def parse_seq(
    df: DataFrame, input_col: str, prefix: str, compute_set: bool = False
) -> DataFrame:
    """
    Parse an encoded sequence column into historical and next-step arrays.

    The input column is expected to contain space-separated sequences,
    where each sequence contains comma-separated integer values. The
    final sequence is treated as the next sequence and all preceding
    sequences are treated as history.

    Parameters
    ----------
    df
        Input Spark DataFrame containing the encoded sequence column.
    input_col
        Name of the column containing the encoded sequence string.
    prefix
        Prefix used for the generated column names.
    compute_set
        Whether to additionally create distinct flattened sets for the
        historical and next sequences.

    Returns
    -------
    DataFrame
        DataFrame containing the original columns together with parsed
        raw, historical, next, and integer-array columns. When
        ``compute_set`` is True, distinct historical and next-set
        columns are also included.
    """

    df = (
        df.withColumn(f"{prefix}_raw", F.split(F.col(input_col), " "))
        .withColumn(
            f"{prefix}_prev",
            F.slice(F.col(f"{prefix}_raw"), 1, F.size(f"{prefix}_raw") - 1),
        )
        .withColumn(
            f"{prefix}_all",
            F.expr(
                f"""
                    transform(
                        {prefix}_prev,
                        x -> transform(
                            split(x, '_'), y -> cast(y as int)
                        )
                    )
                """
            ),
        )
        .withColumn(f"{prefix}_next", F.element_at(f"{prefix}_raw", -1))
        .withColumn(
            f"next_{prefix}_int",
            F.expr(
                f"""
                    transform(
                        split({prefix}_next, '_'), x -> cast(x as int)
                    )
                """
            ),
        )
    )

    if compute_set:
        df = df.withColumn(
            f"{prefix}_set", F.array_distinct(F.flatten(F.col(f"{prefix}_all")))
        ).withColumn(
            f"next_{prefix}_set", F.array_distinct(F.col(f"next_{prefix}_int"))
        )

    return df


def filtered_orders(path: str, spark: SparkSession) -> DataFrame:
    data_path = join_path(path, "orders")
    orders = read_parquet(data_path, spark)
    filtered_orders = orders.filter(F.col("eval_set").isin("train", "test")).select(
        "user_id", F.col("eval_set").alias("train_eval_set")
    )
    return filtered_orders


def build_each_product_in_order_history(
    path: str, df: DataFrame, orders: DataFrame, spark: SparkSession
) -> DataFrame:
    df = df.join(orders, how="left", on="user_id")

    df = (
        df.withColumn(
            "order_size_history",
            F.expr(
                """
                    array_join(
                        transform(products_all, x -> size(x)),
                        " "
                    )
                """
            ),
        )
        .withColumn(
            "reorder_size_history",
            F.expr(
                """
                    array_join(
                        transform(
                            sequence(1, size(products_all)),
                            i ->
                            CASE
                            WHEN i = 1 THEN 0 ELSE
                            size(
                                array_intersect(
                                    array_distinct(flatten(slice(
                                        products_all, 1, i-1
                                        )
                                    )
                                ),
                                    element_at(products_all, i)
                                )
                            )
                            END
                        ),
                        " "
                    )
                """
            ),
        )
        .withColumn("product_id", F.explode("products_set"))
        .withColumn(
            "label",
            F.when(
                F.col("train_eval_set") == "train",
                F.array_contains(F.col("next_products_set"), F.col("product_id")).cast(
                    "int"
                ),
            ).otherwise(-1),
        )
        .withColumn(
            "is_ordered_history",
            F.expr(
                """
                    array_join(
                        transform(
                            products_all,
                            x -> cast(array_contains(x, product_id) as int)
                        ),
                        " "
                    )
                """
            ),
        )
        .withColumn(
            "pos_in_order_history",
            F.expr(
                """
                    array_join(
                        transform(
                            products_all,
                            x ->
                            CASE
                            WHEN array_contains(x, product_id)
                            THEN array_position(x, product_id)
                            ELSE 0 END
                        ),
                        " "
                    )
                """
            ),
        )
    )

    products = read_parquet(join_path(path, "products"), spark)
    df = df.join(products, how="left", on="product_id")

    return df.select(SELECTED_COLUMNS)


def build_each_reorder_history(df: DataFrame, orders: DataFrame):
    df = df.join(orders, how="left", on="user_id")

    df = (
        df.withColumn("product_id", F.lit(0))
        .withColumn("aisle_id", F.lit(0))
        .withColumn("department_id", F.lit(0))
        .withColumn("product_name", F.lit(""))
        .withColumn(
            "label",
            F.when(
                F.col("train_eval_set") == "train",
                (F.array_max("next_reorders_int") == 0).cast("int"),
            ).otherwise(-1),
        )
        .withColumn(
            "is_ordered_history",
            F.expr(
                """
                    array_join(
                        transform(
                            reorders_all,
                            x -> cast(array_max(x) == 0 as int)
                        ),
                        " "
                    )
                """
            ),
        )
        .withColumn(
            "position_in_order_history",
            F.expr(
                """
                    array_join(
                        transform(reorders_all, x -> 0),
                        " "
                    )
                """
            ),
        )
        .withColumn(
            "history_order_size",
            F.expr(
                """
                    array_join(
                        transform(reorders_all, x -> size(x)),
                        " "
                    )
                """
            ),
        )
        .withColumn(
            "history_reorder_size",
            F.expr(
                """
                    array_join(
                        transform(
                            reorders_all,
                            x -> aggregate(
                                x, cast(0 as bigint), (acc, v) -> acc + v
                            )
                        ),
                        " "
                    )
                """
            ),
        )
    )

    return df


if __name__ == "__main__":
    from instacart_etl_rnn.common.io import read_parquet
    from instacart_etl_rnn.common.spark import create_spark_session

    spark = create_spark_session("product")
    df = read_parquet("data/gcs/user_data", spark)
    df = parse_seq(df, "reorders", "reorders", False)

    orders = filtered_orders("data/gcs", spark)
    # df = build_each_product_in_order_history("", df, orders, spark)
    df = build_each_reorder_history(df, orders)
    df.select(
        "label",
        "is_ordered_history",
        "position_in_order_history",
        "history_order_size",
        "history_reorder_size",
        "train_eval_set",
    ).show()
