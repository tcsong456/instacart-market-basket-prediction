from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType

from instacart_etl_rnn.common.utils import pad_array, parse_string_sequence


def transform_reorder_size_data(user_data: DataFrame) -> DataFrame:
    """
    Generate order-level reorder features and training labels from user reorder
    histories. Each user's reorder history is split into a history portion and
    a target order. The history is converted from whitespace-delimited reorder
    strings into arrays of integers, from which order sizes and reorder counts are
    derived. The target order is used to generate the training label.

    Args:
        user_data: DataFrame containing user reorder histories, where the
            ``reorders`` column is a whitespace-delimited sequence of
            underscore-delimited reorder indicators (for example,
            ``"1_0_1 0_1"``).

    Returns:
        DataFrame containing the original user information together with:
            - ``reorders_prev``: Historical reorder sequences as
              ``array<array<int>>``.
            - ``reorders_next``: Target reorder sequence as ``array<int>``.
            - ``order_sizes``: Number of products in each historical order.
            - ``reorder_sizes``: Number of reordered products in each
              historical order.
            - ``label``: Total number of reordered products in the target
              order.
    """

    user_data = (
        user_data.withColumn(
            "reorders",
            F.when(
                F.col("reorders").isNull() | (F.trim(F.col("reorders")) == ""),
                F.array().cast(ArrayType(StringType())),
            ).otherwise(F.split(F.trim(F.col("reorders")), r"\s+")),
        )
        .withColumn(
            "reorders_prev",
            F.when(
                (F.size("reorders") > 1),
                F.slice("reorders", 1, F.size(F.col("reorders")) - 1),
            ).otherwise(F.col("reorders")),
        )
        .withColumn(
            "reorders_next",
            F.when((F.size("reorders") > 1), F.element_at("reorders", -1)).otherwise(
                F.lit("")
            ),
        )
        .withColumn(
            "reorders_prev",
            F.transform("reorders_prev", lambda x: parse_string_sequence(x, "_")),
        )
        .withColumn("reorders_next", parse_string_sequence(F.col("reorders_next"), "_"))
        .withColumn(
            "order_sizes",
            F.expr(
                """
                    transform(
                        reorders_prev,
                        x -> cast(size(x) as int)
                    )
                """
            ),
        )
        .withColumn(
            "reorder_sizes",
            F.expr(
                """
                    transform(
                        reorders_prev,
                        x -> cast(aggregate(x, 0, (acc, v) -> acc + v) as int)
                    )
                """
            ),
        )
        .withColumn(
            "label",
            F.expr(
                """
                    aggregate(
                        reorders_next,
                        0,
                        (acc, v) -> acc + v
                    )
                """
            ),
        )
    )

    return user_data


def pad_column_arrays(df: DataFrame, pad_length: int = 30) -> DataFrame:
    """
    Parse sequential feature columns and pad them to a fixed length.
    Temporal feature columns are first converted from whitespace-delimited
    strings into integer arrays. All temporal and order-size feature arrays are
    then truncated or padded with zeros to the specified length. The original
    (truncated) length of the reorder history is stored in ``history_length``.

    Args:
        df: DataFrame containing sequential user history features.
        pad_length: Target length for all output arrays. Arrays longer than
            this value are truncated, while shorter arrays are padded with
            zeros.

    Returns:
        DataFrame with parsed and fixed-length array features.
    """

    TEMPORAL_COLUMNS = [
        "order_numbers",
        "order_dows",
        "order_hours",
        "days_since_prior_orders",
    ]
    SIZE_COLUMNS = [
        "order_sizes",
        "reorder_sizes",
    ]
    TOTAL_COLUMNS = TEMPORAL_COLUMNS + SIZE_COLUMNS

    for colname in TEMPORAL_COLUMNS:
        if colname == "days_since_prior_orders":
            df = df.withColumn(
                colname, parse_string_sequence(F.col(colname), data_type="double")
            )
        else:
            df = df.withColumn(colname, parse_string_sequence(F.col(colname)))

    for colname in TOTAL_COLUMNS:
        if colname == "days_since_prior_orders":
            padded_array, padded_length = pad_array(
                F.col(colname), pad_length, "double"
            )
        else:
            padded_array, padded_length = pad_array(F.col(colname), pad_length)

        if colname == "reorder_sizes":
            df = df.withColumn("history_length", padded_length)

        df = df.withColumn(colname, padded_array)

    return df.select(
        "user_id",
        "eval_set",
        "order_sizes",
        "reorder_sizes",
        "label",
        "order_dows",
        "order_hours",
        "days_since_prior_orders",
        "order_numbers",
        "history_length",
    )
