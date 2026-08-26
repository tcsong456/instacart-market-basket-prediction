from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.common.paths import join_path
from instacart_etl_rnn.simulation.create_user_split import (
    add_order_role,
    build_order_simulation_split,
    build_user_simulation_split,
)
from instacart_etl_rnn.validation.dataset import validate_dataset
from instacart_etl_rnn.validation.loader import load_contract


def run_user_split_job(
    spark: SparkSession,
    input_path: str,
    output_path: str,
    contract_path: str,
    period: str,
) -> None:
    orders = read_parquet(join_path(input_path, "orders"), spark)
    orders = orders.filter(F.col("eval_set").isin("prior", "train"))

    user_split = build_user_simulation_split(
        orders=orders,
    )

    order_split = build_order_simulation_split(orders=orders, user_split=user_split)

    available_orders = add_order_role(order_split, period)
    available_orders.persist(storageLevel=StorageLevel.MEMORY_AND_DISK)

    try:
        contract = load_contract(join_path(contract_path, "user_split_data.yaml"))
        validate_dataset(available_orders, contract=contract)

        write_parquet(join_path(output_path, "available_orders"), available_orders)
    finally:
        available_orders.unpersist()
