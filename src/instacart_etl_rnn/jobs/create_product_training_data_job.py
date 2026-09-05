from pyspark import StorageLevel
from pyspark.sql import SparkSession

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.common.paths import join_path
from instacart_etl_rnn.gold.create_product_training_data import (
    build_product_training_data,
    build_word_idx,
    encode_product_names,
)
from instacart_etl_rnn.validation.dataset import validate_dataset
from instacart_etl_rnn.validation.loader import load_contract


def run_product_training_data_job(
    spark: SparkSession,
    raw_path: str,
    input_path: str,
    output_path: str,
    contract_path: str,
    mode: str,
    min_word_freq: int = 5,
    product_name_length: int = 30,
    encode_length: int = 100,
) -> None:
    """Run the product sequence training-data pipeline.

    Reads product metadata and product history data, builds the product-name
    vocabulary and encoded product names, constructs the model-ready training
    dataset, and writes the result to the output location.

    Args:
        spark: Active Spark session.
        raw_path: Base path containing the products dataset.
        input_path: Base path containing product history data.
        output_path: Base path where product training data will be written.
        min_word_freq: Minimum word frequency required for a token to receive
            a positive vocabulary index.
        product_name_length: Maximum encoded product-name sequence length.
        encode_length: Maximum historical feature sequence length.

    Returns:
        None.
    """

    products = read_parquet(join_path(raw_path, "products"), spark)
    product_history_data = read_parquet(
        join_path(input_path, f"product_history_data_{mode}"), spark
    )

    word_index = build_word_idx(products, min_word_freq)
    word_index = word_index.persist(StorageLevel.MEMORY_AND_DISK)
    word_index.count()

    encoded_product_name = encode_product_names(products, word_index)
    encoded_product_name = encoded_product_name.persist(StorageLevel.MEMORY_AND_DISK)
    encoded_product_name.count()

    product_training_data = build_product_training_data(
        product_history_data=product_history_data,
        encoded_product_name=encoded_product_name,
        product_name_length=product_name_length,
        encode_length=encode_length,
    )
    product_training_data.persist(StorageLevel.MEMORY_AND_DISK)

    try:
        contract = load_contract(join_path(contract_path, "product_training_data.yaml"))
        validate_dataset(product_training_data, contract=contract)

        write_parquet(
            join_path(output_path, f"product_training_data_{mode}"),
            product_training_data,
        )
    finally:
        product_training_data.unpersist()
        encoded_product_name.unpersist()
        word_index.unpersist()
