from pyspark.sql import SparkSession

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.common.paths import join_path
from instacart_etl_rnn.gold.create_aisle_history_data import (
    build_aisle_history_data,
    parse_seq,
)

SELECTED_COLUMNS = [
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
]


def run_aisle_history_job(
    spark: SparkSession, input_path: str, data_path: str, output_path: str
) -> None:
    user_data = read_parquet(join_path(input_path, "user_data"), spark)

    df = parse_seq(user_data)
    df = build_aisle_history_data(df)

    products = read_parquet(join_path(data_path, "products"), spark)

    aisle_history_data = df.join(
        products.select("aisle_id", "department_id").distinct(),
        how="left",
        on="aisle_id",
    ).select(SELECTED_COLUMNS)

    write_parquet(join_path(output_path, "aisle_history_data"), aisle_history_data)
