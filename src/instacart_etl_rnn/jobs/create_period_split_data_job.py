import copy

from pyspark.sql import SparkSession

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.common.paths import join_path
from instacart_etl_rnn.simulation.create_order_product_split import (
    select_base_model_users,
    select_stacking_model_users,
    split_order_products_by_role,
)
from instacart_etl_rnn.validation.dataset import validate_dataset
from instacart_etl_rnn.validation.loader import load_contract

COLUMNS = [
    "product_id",
    "order_id",
    "add_to_cart_order",
    "reordered",
    "user_id",
    "eval_set",
    "order_number",
    "order_dow",
    "order_hour_of_day",
    "days_since_prior_order",
    "product_name",
    "aisle_id",
    "department_id",
]


def run_order_products_split_job(
    spark: SparkSession,
    input_path: str,
    output_path: str,
    mode: str,
    period: str,
    contract_path: str,
) -> None:
    """Build and persist order-product splits for model training.

    Reads the simulated order-product dataset, selects users for either the
    base-model or stacking-model workflow, and splits the selected data into
    training, validation, and evaluation histories based on availability
    flags.

    Base-model mode writes training, validation, and evaluation datasets for
    the requested simulation period. Stacking mode writes only training and
    validation datasets under the ``stacking_train`` output path.

    Each output is restricted to the persisted split columns and validated
    against the base order-products split contract before being written.

    Args:
        spark: Active Spark session.
        input_path: Base path containing the order-products input dataset.
        output_path: Base path for persisted split datasets.
        mode: Model workflow to build. Must be ``base_train`` or
            ``stacking_train``.
        period: Simulation period. Must be ``initial``, ``t1``, or ``t2``.
        contract_path: Base path containing the split data contract.

    Raises:
        ValueError: If ``mode`` or ``period`` is unsupported.
    """

    if mode not in ["base_train", "stacking_train"]:
        raise ValueError(
            f"mode must be either base_train or stacking_train, but received {mode}"
        )

    if period not in ["initial", "t1", "t2"]:
        raise ValueError(
            f"period can only be one of [initial, t1, t2], but received {period}"
        )

    order_products = read_parquet(join_path(input_path, "order_products"), spark)

    if mode == "base_train":
        model = select_base_model_users(order_products)
    else:
        period = "stacking_train"
        model = select_stacking_model_users(order_products)

    train_history, evaluation_history, validation_history = (
        split_order_products_by_role(model)
    )

    base_contract = load_contract(
        join_path(
            contract_path,
            "order_products_split_base.yaml",
        )
    )

    role_datasets = {
        "order_products_train": train_history,
        "order_products_validation": validation_history,
    }

    if mode == "base_train":
        role_datasets["order_products_evaluation"] = evaluation_history

    for name, df in role_datasets.items():
        contract = copy.deepcopy(base_contract)
        contract["dataset"]["name"] = name

        output_df = df.select(COLUMNS)

        validate_dataset(
            output_df,
            contract=contract,
        )

        write_parquet(
            join_path(f"{output_path}/{period}", name),
            output_df,
        )
