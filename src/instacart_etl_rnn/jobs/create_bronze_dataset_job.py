from pathlib import Path

from pyspark.sql import SparkSession

from instacart_etl_rnn.bronze.create_bronze_dataset import (
    build_dependent_bronze_datasets,
    build_independent_bronze_datasets,
)


def run_bronze_job(
    spark: SparkSession,
    *,
    csv_path: str,
    parquet_path: str,
    contract_path: Path,
) -> None:
    build_independent_bronze_datasets(
        spark,
        csv_path=csv_path,
        parquet_path=parquet_path,
        contract_path=contract_path,
    )

    build_dependent_bronze_datasets(
        spark,
        csv_path=csv_path,
        parquet_path=parquet_path,
        contract_path=contract_path,
    )
