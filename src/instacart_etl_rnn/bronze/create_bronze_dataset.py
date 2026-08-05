import logging
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from instacart_etl_rnn.common.io import read_csv, read_parquet, write_parquet
from instacart_etl_rnn.common.paths import PathLike, join_path
from instacart_etl_rnn.validation.dataset import validate_dataset
from instacart_etl_rnn.validation.loader import load_contract

logger = logging.getLogger(__name__)


_CONTRACT_TYPE_MAP = {
    "integer": IntegerType(),
    "long": LongType(),
    "string": StringType(),
    "double": DoubleType(),
    "boolean": BooleanType(),
}


def build_spark_schema(contract: dict[str, Any]) -> StructType:
    """
    Build a Spark schema from a data contract.

    Parameters
    ----------
    contract
        Data contract containing a ``schema`` section that defines
        column names, data types, and nullability.

    Returns
    -------
    StructType
        Spark schema corresponding to the contract.

    Raises
    ------
    ValueError
        If the contract contains an unsupported data type.
    """

    fields = []
    for schema in contract["schema"]:
        contract_type = schema["type"]
        null = schema["nullable"]
        column_name = schema["name"]

        sparktype = _CONTRACT_TYPE_MAP.get(contract_type)
        if sparktype is None:
            raise ValueError(
                f"Unsupported contract type {contract_type!r} "
                f"for column {column_name!r}"
            )

        fields.append(StructField(column_name, sparktype, null))

    return StructType(fields)


def convert_csv_to_parquet(
    spark: SparkSession,
    *,
    input_path: PathLike,
    output_path: PathLike,
    contract: dict[str, Any],
    reference_datasets: dict[str, DataFrame] | None = None,
) -> None:
    """
    Read, validate, and convert a CSV dataset to Parquet.

    The CSV file is read using a Spark schema built from the data
    contract. The resulting DataFrame is validated against the contract
    before being written to the specified Parquet output path.

    Parameters
    ----------
    spark
        Active Spark session used to read and write the dataset.
    input_path
        Path to the source CSV file.
    output_path
        Destination path for the Parquet dataset.
    contract
        Data contract defining the dataset name, schema, constraints,
        relationships, and business rules.
    reference_datasets
        Optional mapping of reference dataset names to DataFrames used
        for referential-integrity validation.

    Raises
    ------
    ValueError
        If the contract contains an unsupported data type.
    DataValidationError
        If the dataset violates a contract rule configured to fail the
        validation process.
    """

    dataset_config = contract["dataset"]
    dataset_name = dataset_config["name"]

    schema = build_spark_schema(contract)

    logger.info("Reading raw csv dataset %s from %s", dataset_name, input_path)

    df = read_csv(path=input_path, spark=spark, schema=schema)

    logger.info("validating dataset %s against its contract", dataset_name)

    validate_dataset(df, contract=contract, reference_datasets=reference_datasets)

    logger.info(
        "Validation completed for dataset '%s'; writing Parquet to %s",
        dataset_name,
        output_path,
    )

    write_parquet(path=output_path, df=df)

    logger.info(
        "Bronze ingestion completed for dataset '%s'",
        dataset_name,
    )


def build_independent_bronze_datasets(
    spark: SparkSession, *, input_path: str, output_path: str, contract_path: Path
) -> None:
    """
    Build independent bronze datasets from raw CSV files.

    The function processes the orders, aisles, and departments
    datasets. For each dataset, it builds the source CSV path,
    destination Parquet path, and contract path, loads the contract,
    and delegates CSV reading, validation, and Parquet writing to
    ``convert_csv_to_parquet``.

    Parameters
    ----------
    spark
        Active Spark session used to process the datasets.
    input_path
        Base path containing the raw CSV files.
    output_path
        Base path where the bronze Parquet datasets are written.
    contract_path
        Directory containing the dataset contract YAML files.

    Raises
    ------
    FileNotFoundError
        If a required contract file does not exist.
    InvalidContractError
        If a loaded contract is invalid.
    DataValidationError
        If one of the datasets fails contract validation.
    """

    dataset_names = ["orders", "aisles", "departments"]
    for dataset_name in dataset_names:
        data_csv_path = join_path(input_path, f"{dataset_name}.csv")
        data_parquet_path = join_path(output_path, dataset_name)
        contract_dir = contract_path / f"{dataset_name}.yaml"
        contract = load_contract(contract_dir)

        convert_csv_to_parquet(
            spark=spark,
            input_path=data_csv_path,
            output_path=data_parquet_path,
            contract=contract,
        )


def build_dependent_bronze_datasets(
    spark: SparkSession, *, parquet_path: str, csv_path: str, contract_path: Path
) -> None:
    """
    Build bronze datasets that depend on reference datasets.

    The function first builds the products dataset using the existing
    aisles and departments bronze datasets as foreign-key references.
    It then builds the prior and train order-products datasets using
    the existing orders and newly created products bronze datasets as
    references.

    Parameters
    ----------
    spark
        Active Spark session used to read and write datasets.
    parquet_path
        Base path containing existing bronze Parquet datasets and where
        dependent bronze datasets will be written.
    csv_path
        Base path containing the source CSV files.
    contract_path
        Directory containing the products and order-products contract
        YAML files.

    Raises
    ------
    FileNotFoundError
        If a required contract or input dataset cannot be found.
    InvalidContractError
        If a loaded data contract is invalid.
    DataValidationError
        If a dataset fails validation against its data contract.
    """

    products_contract = load_contract(contract_path / "products.yaml")
    order_products_contract = load_contract(contract_path / "order_products.yaml")

    aisles = read_parquet(join_path(parquet_path, "aisles"), spark)

    departments = read_parquet(join_path(parquet_path, "departments"), spark)

    convert_csv_to_parquet(
        spark,
        input_path=join_path(csv_path, "products.csv"),
        output_path=join_path(parquet_path, "products"),
        contract=products_contract,
        reference_datasets={"aisles": aisles, "departments": departments},
    )

    orders = read_parquet(join_path(parquet_path, "orders"), spark)

    products = read_parquet(join_path(parquet_path, "products"), spark)

    convert_csv_to_parquet(
        spark,
        input_path=join_path(csv_path, "order_products__prior.csv"),
        output_path=join_path(parquet_path, "order_products__prior"),
        contract=order_products_contract,
        reference_datasets={"orders": orders, "products": products},
    )

    convert_csv_to_parquet(
        spark,
        input_path=join_path(csv_path, "order_products__train.csv"),
        output_path=join_path(parquet_path, "order_products__train"),
        contract=order_products_contract,
        reference_datasets={"orders": orders, "products": products},
    )
