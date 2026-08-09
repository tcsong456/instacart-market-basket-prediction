import logging

from pyspark.sql import SparkSession

from instacart_etl_rnn.common.io import write_parquet
from instacart_etl_rnn.silver.create_user_data import (
    build_order_level_data,
    build_user_level_data,
    read_input_datasets,
)

logger = logging.getLogger(__name__)


def run_user_data_job(
    spark: SparkSession,
    path: str,
    contract_path: str,
) -> None:
    logger.info("Reading silver order_products dataset")

    order_products = read_input_datasets(
        spark=spark, input_path=path, contract_path=contract_path
    )

    logger.info("Builing order level data")

    order_group_data = build_order_level_data(order_products)

    logger.info("Building user data")

    user_data = build_user_level_data(order_group_data)

    logger.info(f"Writing user data to {path}")

    write_parquet(f"{path}/user_data", user_data)
