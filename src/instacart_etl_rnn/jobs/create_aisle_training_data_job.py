from pyspark import StorageLevel
from pyspark.sql import SparkSession

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.common.paths import join_path
from instacart_etl_rnn.gold.create_aisle_training_data import parse_aisle_seq_data
from instacart_etl_rnn.validation.dataset import validate_dataset
from instacart_etl_rnn.validation.loader import load_contract


def run_aisle_training_data_job(
    spark: SparkSession,
    input_path: str,
    output_path: str,
    mode: str,
    contract_path: str,
    pad_length: int = 30,
) -> None:
    aisle_history_data = read_parquet(
        join_path(input_path, f"aisle_history_data_{mode}"), spark
    )
    aisle_training_data = parse_aisle_seq_data(aisle_history_data, pad_length)
    aisle_training_data.persist(StorageLevel.MEMORY_AND_DISK)

    try:
        contract = load_contract(join_path(contract_path, "aisle_training_data.yaml"))
        validate_dataset(aisle_training_data, contract=contract)

        write_parquet(
            join_path(output_path, f"aisle_training_data_{mode}"), aisle_training_data
        )
    finally:
        aisle_training_data.unpersist()
