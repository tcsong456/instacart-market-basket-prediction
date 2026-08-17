from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def parse_seq(df: DataFrame) -> DataFrame:
    """
    Parse sequential aisle history strings into structured array features.
    The input ``aisle_ids`` column contains a space-separated sequence of
    orders, where each order is represented by underscore-separated aisle
    IDs. This function converts the raw string representation into array-based
    features for downstream feature engineering.

    Args:
        df: Input DataFrame containing an ``aisle_ids`` column.

    Returns:
        DataFrame with the parsed aisle history features appended.
    """

    df = (
        df.withColumn("aisle_raw", F.split("aisle_ids", " "))
        .withColumn(
            "aisle_prev",
            F.when(F.size("aisle_raw") == 1, F.col("aisle_raw")).otherwise(
                F.slice("aisle_raw", 1, F.size("aisle_raw") - 1)
            ),
        )
        .withColumn(
            "aisle_all",
            F.expr(
                """
                    transform(
                        aisle_prev,
                        x -> transform(
                            split(x, '_'),
                            y -> cast(y as int)
                        )
                    )
                """
            ),
        )
        .withColumn("aisle_set", F.array_distinct(F.flatten("aisle_all")))
    )

    return df


def build_aisle_history_data(df: DataFrame) -> DataFrame:
    """
    Generate per-aisle historical features from parsed aisle histories.
    Expands each user's historical aisle set into one row per aisle and
    constructs sequential features describing the user's interaction with
    that aisle across previous orders.

    Args:
        df: Input DataFrame produced by ``parse_seq()``, containing the
            parsed aisle history columns (``aisle_all``, ``aisle_set``,
            etc.).

    Returns:
        DataFrame with one row per user-aisle pair and the corresponding
        historical aisle features appended.
    """

    df = (
        df.withColumn(
            "aisle_history_size",
            F.expr(
                """
                    array_join(
                        transform(
                            aisle_all,
                            x -> size(array_distinct(x))
                        ),
                        " "
                    )
                """
            ),
        )
        .withColumn("aisle_id", F.explode("aisle_set"))
        .withColumn(
            "is_ordered_history",
            F.expr(
                """
                    array_join(
                        transform(
                            aisle_all,
                            x -> cast(array_contains(x, aisle_id) as int)
                        ),
                        " "
                    )
                """
            ),
        )
        .withColumn(
            "position_in_order",
            F.expr(
                """
                    array_join(
                        transform(
                            aisle_all,
                            x ->
                            CASE
                            WHEN array_contains(x, aisle_id)
                            THEN array_position(array_distinct(x), aisle_id)
                            ELSE 0
                            END
                        ),
                        " "
                    )
                """
            ),
        )
        .withColumn(
            "num_products_from_aisle",
            F.expr(
                """
                    array_join(
                        transform(
                            aisle_all,
                            x -> size(filter(x, y -> y = aisle_id))
                        ),
                        " "
                    )
                """
            ),
        )
    )

    return df
