from pyspark.sql import SparkSession

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.common.paths import join_path
from instacart_etl_rnn.gold.create_product_history_data import (
    build_each_product_in_order_history,
    build_each_reorder_history,
    filtered_orders,
    parse_seq,
)
from instacart_etl_rnn.validation.dataset import validate_dataset
from instacart_etl_rnn.validation.loader import load_contract


def run_product_history_job(
    spark: SparkSession,
    input_path: str,
    data_path: str,
    output_path: str,
    contract_path: str,
):
    user_data = read_parquet(join_path(input_path, "user_data"), spark)

    orders = filtered_orders(data_path, spark)

    products = parse_seq(user_data, "product_ids", "products", True)
    reorders = parse_seq(user_data, "reorders", "reorders")

    product_history = build_each_product_in_order_history(
        df=products, path=data_path, orders=orders, spark=spark
    )
    reorder_history = build_each_reorder_history(reorders, orders)
    product_include_none_history = product_history.unionByName(reorder_history)

    contract = load_contract(join_path(contract_path, "product_history_data.yaml"))
    validate_dataset(product_include_none_history, contract=contract)

    write_parquet(
        join_path(output_path, "product_history_data"), product_include_none_history
    )
