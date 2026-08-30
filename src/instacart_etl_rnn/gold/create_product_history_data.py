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
    where each sequence contains underscore-separated integer values. The
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
            F.when(
                F.size(F.col(f"{prefix}_raw")) == 1,
                F.col(f"{prefix}_raw"),
            ).otherwise(
                F.slice(
                    F.col(f"{prefix}_raw"),
                    1,
                    F.size(F.col(f"{prefix}_raw")) - 1,
                )
            ),
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
        .withColumn(
            f"{prefix}_next",
            F.when(
                F.size(F.col(f"{prefix}_raw")) == 1,
                F.array().cast("array<string>"),
            ).otherwise(
                F.array(
                    F.element_at(
                        F.col(f"{prefix}_raw"),
                        -1,
                    )
                )
            ),
        )
        .withColumn(
            f"next_{prefix}_int",
            F.expr(
                f"""
                flatten(
                    transform(
                        {prefix}_next,
                        x -> transform(
                            split(x, '_'),
                            y -> cast(y as int)
                        )
                    )
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


def build_each_product_in_order_history(
    path: str, df: DataFrame, spark: SparkSession
) -> DataFrame:
    """Build candidate-product history features and next-order labels.

    Explodes historical candidate products into one row per product and
    creates sequence features describing the product's order history.

    The final supplied order is treated as the prediction target through
    ``next_products_set``. The resulting ``label`` indicates whether each
    candidate product appears in that target order.

    Args:
        path: Base path containing the products dataset.
        df: User-level product sequence DataFrame.
        spark: Active Spark session.

    Returns:
        Product-level DataFrame containing historical sequence features,
        product metadata, and the binary next-order label.
    """

    df = (
        df.withColumn(
            "history_order_size",
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
            "history_reorder_size",
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
            F.array_contains(F.col("next_products_set"), F.col("product_id")).cast(
                "int"
            ),
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
            "position_in_order_history",
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


def build_each_reorder_history(df: DataFrame):
    """Build history features and labels for the no-reorder candidate.

    Creates one synthetic candidate per user representing the case where
    none of the previously ordered products are reordered in the target
    order. The synthetic candidate uses ``product_id = 0`` and zero/default
    product metadata.

    The label is 1 when the target reorder indicators contain no reordered
    products, and 0 otherwise. Historical reorder sequences are converted
    into binary ordered-history indicators, order sizes, reorder counts,
    and zero-valued product positions.

    Args:
        df: User-level reorder sequence DataFrame containing historical
            reorder arrays and the target-order reorder indicators.

    Returns:
        DataFrame containing the synthetic no-reorder candidate with its
        historical sequence features and binary target label.
    """

    df = (
        df.withColumn("product_id", F.lit(0))
        .withColumn("aisle_id", F.lit(0))
        .withColumn("department_id", F.lit(0))
        .withColumn("product_name", F.lit(""))
        .withColumn(
            "label",
            (F.array_max("next_reorders_int") == 0).cast("int"),
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

    return df.select(SELECTED_COLUMNS)
