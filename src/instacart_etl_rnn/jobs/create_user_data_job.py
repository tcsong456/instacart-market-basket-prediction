import logging

from pyspark import StorageLevel
from pyspark.sql import SparkSession

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.common.paths import join_path
from instacart_etl_rnn.silver.create_user_data import (
    build_order_level_data,
    build_user_level_data,
)
from instacart_etl_rnn.validation.dataset import validate_dataset
from instacart_etl_rnn.validation.loader import load_contract

logger = logging.getLogger(__name__)


def run_user_data_job(
    spark: SparkSession, path: str, contract_path: str, mode: str
) -> None:
    logger.info("Reading silver order_products dataset")

    if mode not in ["train", "validation", "evaluation"]:
        raise ValueError(
            "mode must be either train or validation or evaluation, "
            f"but received {mode}"
        )

    order_products = read_parquet(join_path(path, f"order_products_{mode}"), spark)

    logger.info("Building order level data")

    order_group_data = build_order_level_data(order_products)

    logger.info("Building user data")

    user_data = build_user_level_data(order_group_data)
    user_data.persist(StorageLevel.MEMORY_AND_DISK)

    try:
        contract = load_contract(join_path(contract_path, "user_data.yaml"))
        validate_dataset(user_data, contract=contract)

        logger.info(f"Writing user data to {path}")

        write_parquet(join_path(path, f"user_data_{mode}"), user_data)
    finally:
        user_data.unpersist()
