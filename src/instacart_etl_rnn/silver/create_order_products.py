import logging

from pyspark.sql import DataFrame, SparkSession

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.common.paths import join_path
from instacart_etl_rnn.validation.dataset import validate_dataset
from instacart_etl_rnn.validation.loader import load_contract

logger = logging.getLogger(__name__)


DATASETS = [
    "products",
    "orders",
    "order_products__prior",
    "order_products__train",
]


def read_input_datasets(spark: SparkSession, input_path: str) -> dict[str, DataFrame]:
    datasets_mapping = {}
    for dataset_name in DATASETS:
        logger.info(
            ("reading bronze parquet dataset %s for building order_products"),
            dataset_name,
        )

        df = read_parquet(join_path(input_path, dataset_name), spark)
        datasets_mapping[dataset_name] = df

    return datasets_mapping


def build_order_products(
    spark: SparkSession, input_path: str, output_path: str, contract_path: str
) -> None:
    """Build, validate, and write the silver order_products dataset.

    Reads the required bronze datasets, combines the prior and train
    order-product records, and enriches them with order and product
    attributes. Missing days_since_prior_order values are replaced with
    -1 before the resulting dataset is validated against its silver-layer
    contract and written as Parquet.

    Args:
        spark: Active Spark session.
        input_path: Base path containing the bronze input datasets.
        output_path: Path where the silver order_products dataset is written.
        contract_path: Base path containing the dataset contracts.

    Returns:
        None.
    """

    df_dicts = read_input_datasets(
        spark=spark,
        input_path=input_path,
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

    contract = load_contract(join_path(contract_path, "order_products_silver.yaml"))
    validate_dataset(order_products, contract=contract)

    write_parquet(path=output_path, df=order_products)
