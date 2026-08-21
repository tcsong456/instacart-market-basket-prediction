from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from instacart_etl_rnn.common.utils import pad_array, parse_string_sequence

PARSE_COLUMNS = [
    "is_ordered_history",
    "position_in_order",
    "num_products_from_aisle",
    "aisle_history_size",
    "order_dows",
    "order_hours",
    "days_since_prior_orders",
    "order_numbers",
]


def parse_aisle_seq_data(df: DataFrame, max_padded_length: int = 30) -> DataFrame:
    for colname in PARSE_COLUMNS:
        if colname == "days_since_prior_orders":
            df = df.withColumn(
                colname, parse_string_sequence(F.col(colname), data_type="double")
            )
            padded_array, padded_length = pad_array(
                F.col(colname), max_padded_length, "double"
            )
        else:
            df = df.withColumn(colname, parse_string_sequence(F.col(colname)))
            padded_array, padded_length = pad_array(F.col(colname), max_padded_length)

        if colname == "is_ordered_history":
            df = df.withColumn("history_length", padded_length)

        df = df.withColumn(colname, padded_array)

    return df.select(
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
    )
