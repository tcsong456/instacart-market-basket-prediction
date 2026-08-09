import logging

from pyspark.sql import DataFrame, SparkSession

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.common.paths import join_path
from instacart_etl_rnn.validation.dataset import validate_dataset
from instacart_etl_rnn.validation.loader import load_contract

logger = logging.getLogger(__name__)


def read_input_datasets(
    spark: SparkSession, input_path: str, contract_path: str, validation: bool = True
) -> dict[str, DataFrame]:
    """Read and optionally validate datasets required to build order_products.

    Reads the products, orders, order_products__prior, and
    order_products__train Parquet datasets from the input path.

    When validation is enabled, each dataset's contract is loaded and any
    reference datasets required by relationship constraints are read before
    validating the dataset.

    Args:
        spark: Active Spark session.
        input_path: Base path containing the input Parquet datasets.
        contract_path: Base path containing the dataset contract files.
        validation: Whether to validate each dataset against its contract.
            Defaults to True.

    Returns:
        A mapping from dataset name to its loaded Spark DataFrame.
    """

    DATASET_CONTRACTS_MAPPING = {
        "products": "products.yaml",
        "orders": "orders.yaml",
        "order_products__prior": "order_products.yaml",
        "order_products__train": "order_products.yaml",
    }

    datasets_mapping = {}
    for dataset_name, contract_name in DATASET_CONTRACTS_MAPPING.items():
        logger.info(
            ("reading bronze parquet dataset %s for building order_proudcts"),
            dataset_name,
        )

        df = read_parquet(join_path(input_path, dataset_name), spark)
        datasets_mapping.update({dataset_name: df})

        if not validation:
            continue

        contract = load_contract(join_path(contract_path, contract_name))
        ref_dataset = {}
        if "relationships" in contract:
            for relationship in contract["relationships"]:
                parent_df_name = relationship.get("parent", {}).get("dataset")
                parent_df = read_parquet(join_path(input_path, parent_df_name), spark)

                ref_dataset.update({parent_df_name: parent_df})

        validate_dataset(df, contract=contract, reference_datasets=ref_dataset)

    return datasets_mapping


def build_order_products(
    spark: SparkSession,
    input_path: str,
    contract_path: str,
    output_path: str,
    validation: bool = True,
) -> None:
    """Build and write the order_products dataset.

    Reads the required input datasets, optionally validates them, combines
    prior and training order-product records, and enriches them with order
    and product information. Missing days_since_prior_order values are
    replaced with -1 before the resulting dataset is written to Parquet.

    Args:
        spark: Active Spark session.
        input_path: Base path containing the input Parquet datasets.
        contract_path: Base path containing the dataset contract files.
        output_path: Path where the resulting order_products dataset is written.
        validation: Whether to validate input datasets before processing.
            Defaults to True.
    """

    df_dicts = read_input_datasets(
        spark=spark,
        input_path=input_path,
        contract_path=contract_path,
        validation=validation,
    )

    orders_prior, orders_train = (
        df_dicts["order_products__prior"],
        df_dicts["order_products__train"],
    )
    orders = df_dicts["orders"]
    products = df_dicts["products"]

    logger.info("Building order_products dataset")
    order_products = orders_prior.unionByName(orders_train)
    order_products = order_products.join(orders, how="left", on=["order_id"]).join(
        products, how="left", on=["product_id"]
    )
    order_products = order_products.fillna({"days_since_prior_order": -1})

    write_parquet(path=output_path, df=order_products)
