from pyspark import StorageLevel
from pyspark.sql import SparkSession

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.common.paths import join_path
from instacart_etl_rnn.gold.create_reorder_size_training_data import (
    pad_column_arrays,
    transform_reorder_size_data,
)
from instacart_etl_rnn.validation.dataset import validate_dataset
from instacart_etl_rnn.validation.loader import load_contract


def run_reorder_size_training_data(
    spark: SparkSession,
    input_path: str,
    output_path: str,
    contract_path: str,
    pad_length: int,
):
    user_data = read_parquet(join_path(input_path, "user_data"), spark)

    df = transform_reorder_size_data(user_data)
    reorder_size_data = pad_column_arrays(df, pad_length)
    reorder_size_data.persist(StorageLevel.MEMORY_AND_DISK)

    try:
        contract = load_contract(
            join_path(contract_path, "reorder_size_training_data.yaml")
        )
        validate_dataset(reorder_size_data, contract=contract)

        write_parquet(
            join_path(output_path, "reorder_size_training_data"), reorder_size_data
        )
    finally:
        reorder_size_data.unpersist()
