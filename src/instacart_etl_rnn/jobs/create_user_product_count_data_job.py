from pyspark.sql import SparkSession

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.common.paths import join_path
from instacart_etl_rnn.gold.create_user_product_count_data import (
    build_user_product_count,
)
from instacart_etl_rnn.validation.dataset import validate_dataset
from instacart_etl_rnn.validation.loader import load_contract


def run_user_product_count_job(
    spark: SparkSession,
    input_path: str,
    output_path: str,
    contract_path: str,
    mode: str,
) -> None:
    if mode not in ["train", "validation", "evaluation"]:
        raise ValueError(
            "mode must be either train or validation or evaluation, "
            f"but received {mode}"
        )

    order_products = read_parquet(
        join_path(input_path, f"order_products_{mode}"), spark
    )

    user_product_count = build_user_product_count(order_products)

    contract = load_contract(join_path(contract_path, "user_product_count.yaml"))

    validate_dataset(user_product_count, contract=contract)

    write_parquet(
        join_path(output_path, f"user_product_count_{mode}"), user_product_count
    )
