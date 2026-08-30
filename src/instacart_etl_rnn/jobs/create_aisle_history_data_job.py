from pyspark import StorageLevel
from pyspark.sql import SparkSession

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.common.paths import join_path
from instacart_etl_rnn.gold.create_aisle_history_data import (
    build_aisle_history_data,
    parse_seq,
)
from instacart_etl_rnn.validation.dataset import validate_dataset
from instacart_etl_rnn.validation.loader import load_contract

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
    spark: SparkSession,
    input_path: str,
    data_path: str,
    output_path: str,
    contract_path: str,
    mode: str,
) -> None:
    user_data = read_parquet(join_path(input_path, f"user_data_{mode}"), spark)

    df = parse_seq(user_data)
    df = build_aisle_history_data(df)

    products = read_parquet(join_path(data_path, "products"), spark)

    aisle_history_data = df.join(
        products.select("aisle_id", "department_id").distinct(),
        how="left",
        on="aisle_id",
    ).select(SELECTED_COLUMNS)
    aisle_history_data.persist(StorageLevel.MEMORY_AND_DISK)

    try:
        contract = load_contract(join_path(contract_path, "aisle_history_data.yaml"))
        validate_dataset(aisle_history_data, contract=contract)

        write_parquet(
            join_path(output_path, f"aisle_history_data_{mode}"), aisle_history_data
        )
    finally:
        aisle_history_data.unpersist()
