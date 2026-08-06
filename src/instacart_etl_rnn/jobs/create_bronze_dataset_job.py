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
    """
    Build all bronze datasets from the raw CSV source.

    The function orchestrates the bronze ingestion workflow by first
    building datasets that have no foreign-key dependencies, followed
    by datasets that depend on previously generated bronze datasets.

    Parameters
    ----------
    spark
        Active Spark session used throughout the bronze ingestion job.
    csv_path
        Base path containing the raw CSV datasets.
    parquet_path
        Base path where the bronze Parquet datasets are written.
    contract_path
        Directory containing the data contract YAML files.

    Raises
    ------
    FileNotFoundError
        If a required contract or input dataset cannot be found.
    InvalidContractError
        If a data contract is invalid.
    DataValidationError
        If a dataset fails validation against its contract.
    """

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
